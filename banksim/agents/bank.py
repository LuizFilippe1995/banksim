from copy import copy

import numpy as np
from mesa import Agent

from banksim.exogeneous_factors import BankSizeDistribution, ExogenousFactors
from banksim.strategies.bank_ewa_strategy import BankEWAStrategy
from banksim.util import Util


class Bank(Agent):

    def __init__(self, bank_size_distribution, is_intelligent, ewa_damping_factor, model):
        super().__init__(Util.get_unique_id(), model)

        self.initialSize = 1 if bank_size_distribution != BankSizeDistribution.LogNormal \
            else Util.get_random_log_normal(-0.5, 1)

        self.interbankHelper = InterbankHelper()
        self.guaranteeHelper = GuaranteeHelper()
        self.depositors = []  # Depositors
        self.LowRiskpoolcorporateClients = []  # Banks choose the total quantity of low risk clients from here
        self.HighRiskpoolcorporateClients = [] # Banks choose the total quantity of high risk clients from here
        
        self.LowRiskcorporateClients = []  # Low risk corporate clients
        self.HighRiskcorporateClients = [] # High risk corporate clients
        
        self.quantityHighRiskcorporateClients = 0
        self.quantityLowRiskcorporateClients = 0
        
        self.risk_appetite = 0 # Proportion of high risk corporate clients in the total quantity of clients
        
        self.liquidityNeeds = 0
        self.bankRunOccurred = False
        self.withdrawalsCounter = 0

        self.balanceSheet = BalanceSheet()
        self.auxBalanceSheet = None

        self.isIntelligent = is_intelligent
        if self.isIntelligent:
            self.strategiesOptionsInformation = BankEWAStrategy.bank_ewa_strategy_list()
            self.currentlyChosenStrategy = None
            self.EWADampingFactor = ewa_damping_factor

    def update_strategy_choice_probability(self):
        list_a = np.array([0.9999 * s.A + s.strategyProfitPercentageDamped for s \
                           in self.strategiesOptionsInformation])
        _exp = np.exp(list_a)
        list_p = _exp / np.sum(_exp)
        list_f = np.cumsum(list_p)
        for i, strategy in enumerate(self.strategiesOptionsInformation):
            strategy.A, strategy.P, strategy.F = list_a[i], list_p[i], list_f[i]

    
    def pick_new_strategy(self):
        probability_threshold = Util.get_random_uniform(1)
        self.currentlyChosenStrategy = [s for s in self.strategiesOptionsInformation if s.F > \
                                            probability_threshold][0]    
            
    def reset(self):
        self.liquidityNeeds = 0
        self.bankRunOccurred = False
        self.withdrawalsCounter = 0
        self.risk_appetite = 0
        
    def reset_collateral(self):
        self.guaranteeHelper = GuaranteeHelper()
    
    def choose_corporateClient(self, strategy=None):
        if strategy is None:
            strategy = self.currentlyChosenStrategy
        self.risk_appetite = strategy.get_gamma_value()
        self.quantityHighRiskcorporateClients = int(ExogenousFactors.numberCorporateClientsPerBank * self.risk_appetite)
        self.quantityLowRiskcorporateClients = ExogenousFactors.numberCorporateClientsPerBank - self.quantityHighRiskcorporateClients
        
        self.HighRiskcorporateClients = self.HighRiskpoolcorporateClients[0:self.quantityHighRiskcorporateClients]
        self.LowRiskcorporateClients = self.LowRiskpoolcorporateClients[0:self.quantityLowRiskcorporateClients]
        
    def setup_balance_sheet_intelligent(self, strategy=None):
        if strategy is None:
            strategy = self.currentlyChosenStrategy
            
        self.balanceSheet.liquidAssets = self.initialSize * strategy.get_beta_value()
        self.balanceSheet.nonFinancialSectorLoanHighRisk = (self.initialSize - self.balanceSheet.liquidAssets)*self.risk_appetite                                                   
        self.balanceSheet.nonFinancialSectorLoanLowRisk = self.initialSize - self.balanceSheet.liquidAssets - self.balanceSheet.nonFinancialSectorLoanHighRisk
        
        self.balanceSheet.interbankLoan = 0
        self.balanceSheet.discountWindowLoan = 0
        self.balanceSheet.deposits = self.initialSize * (strategy.get_alpha_value() - 1)
        self.liquidityNeeds = 0
        self.setup_balance_sheet()        
        
    def setup_balance_sheet(self):
        loan_per_coporate_clientLowRisk = self.balanceSheet.nonFinancialSectorLoanLowRisk/len(self.LowRiskcorporateClients) if len(self.LowRiskcorporateClients)!= 0 else 0
        loan_per_coporate_clientHighRisk = self.balanceSheet.nonFinancialSectorLoanHighRisk/len(self.HighRiskcorporateClients) if len(self.HighRiskcorporateClients)!=0 else 0
        
        for corporateClient in self.LowRiskcorporateClients:
            corporateClient.loanAmount = loan_per_coporate_clientLowRisk
        
        for corporateClient in self.HighRiskcorporateClients:
            corporateClient.loanAmount = loan_per_coporate_clientHighRisk 
            
        deposit_per_depositor = -self.balanceSheet.deposits / len(self.depositors)
        for depositor in self.depositors:
            depositor.make_deposit(deposit_per_depositor)
    
    def get_capital_adequacy_ratio(self):
        if self.is_solvent():
            rwa = self.get_real_sector_risk_weighted_assets()
            total_risk_weighted_assets = self.balanceSheet.liquidAssets * ExogenousFactors.CashRiskWeight + rwa
            
            if self.is_interbank_creditor():
                total_risk_weighted_assets += self.balanceSheet.interbankLoan * ExogenousFactors.InterbankLoanRiskWeight
                
            if total_risk_weighted_assets != 0:
                return -self.balanceSheet.capital / total_risk_weighted_assets
        return 0
    
    def adjust_capital_ratio(self, minimum_capital_ratio_required):
        current_capital_ratio = self.get_capital_adequacy_ratio()
        if current_capital_ratio <= minimum_capital_ratio_required:
            adjustment_factor = current_capital_ratio / minimum_capital_ratio_required
            
            for corporateClient in self.LowRiskcorporateClients:
                original_loan_amount = corporateClient.loanAmount
                new_loan_amount = original_loan_amount * adjustment_factor
                corporateClient.loanAmount = new_loan_amount
                self.balanceSheet.liquidAssets += (original_loan_amount - new_loan_amount)
                
            for corporateClient in self.HighRiskcorporateClients:
                original_loan_amount2 = corporateClient.loanAmount
                new_loan_amount2 = original_loan_amount2 * adjustment_factor
                corporateClient.loanAmount = new_loan_amount2
                self.balanceSheet.liquidAssets += (original_loan_amount2 - new_loan_amount2)

                
            self.update_non_financial_sector_loans()

    def update_non_financial_sector_loans(self):
        self.balanceSheet.nonFinancialSectorLoanLowRisk = sum(
            client.loanAmount for client in self.LowRiskcorporateClients)
        
        self.balanceSheet.nonFinancialSectorLoanHighRisk = sum(
            client.loanAmount for client in self.HighRiskcorporateClients)
        
    def get_real_sector_risk_weighted_assets(self):
        riskLow = self.balanceSheet.nonFinancialSectorLoanLowRisk * ExogenousFactors.LowRiskCorporateLoanRiskWeight
        riskHigh = self.balanceSheet.nonFinancialSectorLoanHighRisk * ExogenousFactors.HighRiskCorporateLoanRiskWeight
        
        return riskLow + riskHigh
        
        
    def withdraw_deposit(self, amount_to_withdraw):
        if amount_to_withdraw > 0:
            self.withdrawalsCounter += 1
        self.liquidityNeeds -= amount_to_withdraw
        return amount_to_withdraw

    def use_liquid_assets_to_pay_depositors_back(self):
        if self.needs_liquidity():
            original_liquid_assets = self.balanceSheet.liquidAssets
            self.liquidityNeeds += original_liquid_assets
            self.balanceSheet.liquidAssets = max(self.liquidityNeeds, 0)
            resulting_liquid_assets = self.balanceSheet.liquidAssets
            total_paid = original_liquid_assets - resulting_liquid_assets
            self.balanceSheet.deposits += total_paid

    def accrue_interest_balance_sheet(self):
        self.balanceSheet.discountWindowLoan *= (1 + ExogenousFactors.centralBankLendingInterestRate)
        self.balanceSheet.liquidAssets *= (1 + self.model.liquidAssetsInterestRate)
        self.calculate_deposits_interest()

    def calculate_deposits_interest(self):
        deposits_interest_rate = 1 + self.model.depositInterestRate
        self.balanceSheet.deposits *= deposits_interest_rate
        for depositor in self.depositors:
            depositor.deposit.amount *= deposits_interest_rate
    
    def collect_loans(self):
        self.balanceSheet.nonFinancialSectorLoanLowRisk = sum(
            client.pay_loan_back() for client in self.LowRiskcorporateClients)
        
        self.balanceSheet.nonFinancialSectorLoanHighRisk = sum(
            client.pay_loan_back() for client in self.HighRiskcorporateClients)
        
        
    def offers_liquidity(self):
        return self.liquidityNeeds > 0

    def needs_liquidity(self):
        return self.liquidityNeeds <= 0

    def is_liquid(self):
        return self.liquidityNeeds >= 0

    def receive_discount_window_loan(self, amount):
        self.balanceSheet.discountWindowLoan = amount
        self.balanceSheet.deposits -= amount
        self.liquidityNeeds -= amount
    
    def use_non_liquid_assets_to_pay_depositors_back(self):
        if self.needs_liquidity():
            total_loans = self.balanceSheet.nonFinancialSectorLoanLowRisk + self.balanceSheet.nonFinancialSectorLoanHighRisk
            liquidity_needed = -self.liquidityNeeds
            total_loans_to_sell = liquidity_needed * (1 + ExogenousFactors.illiquidAssetDiscountRate)
            
            if total_loans > total_loans_to_sell:
                
                # Firstly, banks sell their less risky loans because it is easier to find buyers.
                if self.balanceSheet.nonFinancialSectorLoanLowRisk >= total_loans_to_sell:
                    amount_sold = total_loans_to_sell
                    self.liquidityNeeds = 0
                    self.balanceSheet.deposits += liquidity_needed
                    
                    proportion_of_illiquid_assets_sold = amount_sold / self.balanceSheet.nonFinancialSectorLoanLowRisk
                   
                    for firm in self.LowRiskcorporateClients:
                        firm.loanAmount *= 1 - proportion_of_illiquid_assets_sold
                
                    self.balanceSheet.nonFinancialSectorLoanLowRisk -= amount_sold
                
                # If not enough, banks will sell their total amount of less risky loans and part of (or the totality of) their more risky loans.
                else:
                    x = total_loans_to_sell - self.balanceSheet.nonFinancialSectorLoanLowRisk
                    amount_sold = self.balanceSheet.nonFinancialSectorLoanLowRisk + (self.balanceSheet.nonFinancialSectorLoanHighRisk - x)
                    
                    self.liquidityNeeds = 0
                    self.balanceSheet.deposits += liquidity_needed
                    proportion_of_illiquid_assets_sold_HighRisk = x / self.balanceSheet.nonFinancialSectorLoanHighRisk
                                        
                    for firm in self.LowRiskcorporateClients:
                        firm.loanAmount *= 0
                    for firm in self.HighRiskcorporateClients:
                        firm.loanAmount *= (1 - proportion_of_illiquid_assets_sold_HighRisk)
                        
                    self.balanceSheet.nonFinancialSectorLoanLowRisk = 0
                    self.balanceSheet.nonFinancialSectorLoanHighRisk -= x
                
            else:
                amount_sold = total_loans
                self.liquidityNeeds += amount_sold / (1 + ExogenousFactors.illiquidAssetDiscountRate)
                self.balanceSheet.deposits += liquidity_needed - self.liquidityNeeds
            
                proportion_of_illiquid_assets_sold = amount_sold / total_loans
                
                for firm in self.LowRiskcorporateClients:
                    firm.loanAmount *= 1 - proportion_of_illiquid_assets_sold  
                for firm in self.HighRiskcorporateClients:
                    firm.loanAmount *= 1 - proportion_of_illiquid_assets_sold
            
                self.balanceSheet.nonFinancialSectorLoanLowRisk = 0
                self.balanceSheet.nonFinancialSectorLoanHighRisk = 0
                          
    def get_profit(self):
        resulting_capital = self.balanceSheet.assets + self.balanceSheet.liabilities
        original_capital = self.auxBalanceSheet.assets + self.auxBalanceSheet.liabilities
        if ExogenousFactors.banksHaveLimitedLiability:
            resulting_capital = max(resulting_capital, 0)
        return resulting_capital - original_capital
    
    def calculate_profit(self, minimum_capital_ratio_required):
        if self.isIntelligent:
            strategy = self.currentlyChosenStrategy

            self.bankRunOccurred = (self.withdrawalsCounter > ExogenousFactors.numberDepositorsPerBank / 2)

            if self.bankRunOccurred:
                original_loans = self.auxBalanceSheet.nonFinancialSectorLoanLowRisk + self.auxBalanceSheet.nonFinancialSectorLoanHighRisk
                resulting_loans = self.balanceSheet.nonFinancialSectorLoanLowRisk + self.balanceSheet.nonFinancialSectorLoanHighRisk
                
                delta = original_loans - resulting_loans
                if delta > 0:
                    self.balanceSheet.nonFinancialSectorLoanLowRisk -= (delta * 0.02)
                    self.balanceSheet.nonFinancialSectorLoanHighRisk -= (delta * 0.06)
                    
            profit = self.get_profit()

            strategy.strategyProfit = profit

            if ExogenousFactors.isCapitalRequirementActive:
                current_capital_ratio = self.get_capital_adequacy_ratio()

                if current_capital_ratio < minimum_capital_ratio_required:
                    delta_capital_ratio = minimum_capital_ratio_required - current_capital_ratio
                    strategy.strategyProfit -= delta_capital_ratio

            # Return on Equity, based on initial shareholders equity.
            strategy.strategyProfitPercentage = -strategy.strategyProfit / self.auxBalanceSheet.capital
            strategy.strategyProfitPercentageDamped = strategy.strategyProfitPercentage * self.EWADampingFactor
    
    def liquidate(self):
        #  first, sell assets...
        self.balanceSheet.liquidAssets += (self.balanceSheet.nonFinancialSectorLoanLowRisk + self.balanceSheet.nonFinancialSectorLoanHighRisk)
        self.balanceSheet.nonFinancialSectorLoanLowRisk = 0
        self.balanceSheet.nonFinancialSectorLoanHighRisk = 0
    
        if self.is_interbank_creditor():
            self.balanceSheet.liquidAssets += self.balanceSheet.interbankLoan
            self.balanceSheet.interbankLoan = 0

        # then, resolve liabilities, in order of subordination...

        #  ...1st, discountWindowLoan...
        if self.balanceSheet.liquidAssets > abs(self.balanceSheet.discountWindowLoan):
            self.balanceSheet.liquidAssets += self.balanceSheet.discountWindowLoan
            self.balanceSheet.discountWindowLoan = 0
        else:
            self.balanceSheet.liquidAssets = 0
            self.balanceSheet.discountWindowLoan += self.balanceSheet.liquidAssets

        # ...2nd, interbank loans...
        if self.is_interbank_debtor():
            if self.balanceSheet.liquidAssets > abs(self.balanceSheet.interbankLoan):
                self.balanceSheet.interbankLoan = 0
                self.balanceSheet.liquidAssets += self.balanceSheet.interbankLoan
            else:
                self.balanceSheet.liquidAssets = 0
                self.balanceSheet.interbankLoan += self.balanceSheet.liquidAssets

        # ... finally, if there is any money left, it is proportionally divided among depositors.
        percentage_deposits_payable = self.balanceSheet.liquidAssets / \
        np.absolute(self.balanceSheet.deposits)
        self.balanceSheet.deposits *= percentage_deposits_payable

        for depositor in self.depositors:
            depositor.deposit.amount *= percentage_deposits_payable

        self.balanceSheet.liquidAssets = 0

    def is_insolvent(self):
        return self.balanceSheet.capital > 0

    def is_solvent(self):
        return self.balanceSheet.capital <= 0

    def is_interbank_creditor(self):
        return self.balanceSheet.interbankLoan >= 0

    def is_interbank_debtor(self):
        return self.balanceSheet.interbankLoan < 0
    
    def period_0(self):
        if self.isIntelligent:
            self.update_strategy_choice_probability()
            self.pick_new_strategy()
            self.choose_corporateClient()
            self.setup_balance_sheet_intelligent(self.currentlyChosenStrategy)
        else:
            self.setup_balance_sheet()
        self.auxBalanceSheet = copy(self.balanceSheet)

    def period_1(self):
        # First, banks try to use liquid assets to pay early withdrawals...
        self.use_liquid_assets_to_pay_depositors_back()
        # ... if needed, they will try interbank market by clearing house.
        # ... if banks still needs liquidity, central bank might rescue...

    def period_2(self):
        self.accrue_interest_balance_sheet()
        self.collect_loans()


class InterbankHelper:
    def __init__(self):
        self.counterpartyID = 0
        self.priorityOrder = 0
        self.auxPriorityOrder = 0
        self.loanAmount = 0
        self.acumulatedLiquidity = 0
        self.riskSorting = None
        self.amountLiquidityLeftToBorrowOrLend = 0


class GuaranteeHelper:
    def __init__(self):
        self.potentialCollateral = 0
        self.feasibleCollateral = 0
        self.outstandingAmountImpact = 0
        self.residual = 0
        self.redistributedCollateral = 0
        self.collateralAdjustment = 0


class BalanceSheet:
    def __init__(self):
        self.deposits = 0
        self.discountWindowLoan = 0
        self.interbankLoan = 0
        self.nonFinancialSectorLoanLowRisk = 0
        self.nonFinancialSectorLoanHighRisk = 0
        self.liquidAssets = 0

    @property
    def capital(self):
        return -(self.liquidAssets +
                 self.nonFinancialSectorLoanLowRisk +
                 self.nonFinancialSectorLoanHighRisk +
                 self.interbankLoan +
                 self.discountWindowLoan +
                 self.deposits)

    @property
    def assets(self):
        return self.liquidAssets + self.nonFinancialSectorLoanLowRisk + self.nonFinancialSectorLoanHighRisk + np.max(self.interbankLoan, 0)
    
    @property
    def liabilities(self):
        return self.deposits + self.discountWindowLoan + np.min(self.interbankLoan, 0)
