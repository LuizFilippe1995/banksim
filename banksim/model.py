from mesa import Model

from numba import prange

from banksim.activation import MultiStepActivation
from banksim.agents.bank import Bank
from banksim.agents.central_bank import CentralBank
from banksim.agents.clearing_house import ClearingHouse
from banksim.agents.corporate_client import CorporateClient
from banksim.agents.depositor import Depositor
from banksim.exogeneous_factors import ExogenousFactors, SimulationType, InterbankPriority


class BankingModel(Model):
    """
    BankSim is a banking agent-based simulation framework developed in Python 3+.

    Its main goal is to provide an out-of-the-box simulation tool to study the impacts of a broad range of regulation policies over the banking system.

    The basic model is based on the paper by Barroso, R. V. et al., Interbank network and regulation policies: an analysis through agent-based simulations with adaptive learning, published in the Journal Of Network Theory In Finance, v. 2, n. 4, p. 53–86, 2016.

    The paper is available online at https://mpra.ub.uni-muenchen.de/73308.
    """

    def __init__(self, simulation_type='HighSpread', exogenous_factors=None, number_of_banks=None):
        super().__init__()

        # Simulation data
        self.simulation_type = SimulationType[simulation_type]
        BankingModel.update_exogeneous_factors_by_simulation_type(self.simulation_type)

        BankingModel.update_exogeneous_factors(exogenous_factors, number_of_banks)

        # Economy data
        self.numberBanks = ExogenousFactors.numberBanks
        self.depositInterestRate = ExogenousFactors.depositInterestRate
        self.interbankInterestRate = ExogenousFactors.interbankInterestRate
        self.liquidAssetsInterestRate = ExogenousFactors.liquidAssetsInterestRate
        self.interbankLendingMarketAvailable = ExogenousFactors.interbankLendingMarketAvailable

        # Scheduler
        self.schedule = MultiStepActivation(self)

        # Central Bank
        _params = (ExogenousFactors.centralBankLendingInterestRate,
                   ExogenousFactors.offersDiscountWindowLending,
                   ExogenousFactors.minimumCapitalAdequacyRatio,
                   not ExogenousFactors.isCentralBankZeroIntelligenceAgent,
                   ExogenousFactors.DefaultEWADampingFactor)
        self.schedule.add_central_bank(CentralBank(*_params, self))

        # Clearing House
        _params = (self.numberBanks,
                   ExogenousFactors.isClearingGuaranteeAvailable)
        self.schedule.add_clearing_house(ClearingHouse(*_params, self))

        # Banks
        _params = (ExogenousFactors.bankSizeDistribution,
                   not ExogenousFactors.areBanksZeroIntelligenceAgents,
                   ExogenousFactors.DefaultEWADampingFactor)
        for _ in range(self.numberBanks):
            bank = Bank(*_params, self)
            self.schedule.add_bank(bank)
        self.normalize_banks()

        _params_depositors = (
            not ExogenousFactors.areDepositorsZeroIntelligenceAgents,
            ExogenousFactors.DefaultEWADampingFactor)

        # Depositors and Corporate Clients (Firms)
        
        
        
        _params_corporate_clientsHighRisk = (ExogenousFactors.HighRiskCorporateClientDefaultRate,
                                         ExogenousFactors.HighRiskCorporateClientLossGivenDefault,
                                         ExogenousFactors.HighRiskCorporateClientLoanInterestRate)
        
        _params_corporate_clientsLowRisk = (ExogenousFactors.LowRiskCorporateClientDefaultRate,
                                         ExogenousFactors.LowRiskCorporateClientLossGivenDefault,
                                         ExogenousFactors.LowRiskCorporateClientLoanInterestRate)

        for bank in self.schedule.banks:
            for i in range(ExogenousFactors.numberDepositorsPerBank):
                depositor = Depositor(*_params_depositors, bank, self)
                bank.depositors.append(depositor)
                self.schedule.add_depositor(depositor)
            for i in range(ExogenousFactors.numberCorporateClientsPerBank):
                corporate_client = CorporateClient(*_params_corporate_clientsLowRisk, bank, self)
                bank.LowRiskpoolcorporateClients.append(corporate_client)
                self.schedule.add_corporate_client_LowRisk(corporate_client)
            for i in range(ExogenousFactors.numberCorporateClientsPerBank):
                corporate_client = CorporateClient(*_params_corporate_clientsHighRisk, bank, self)
                bank.HighRiskpoolcorporateClients.append(corporate_client)
                self.schedule.add_corporate_client_HighRisk(corporate_client)

    @jit(parallel = True)       
    def step(self):
        self.schedule.reset_cycle()
        self.schedule.period_0()
        self.schedule.period_1()
        self.schedule.period_2()
        
    
    def run_model(self, n): 
        cycle=0             
        while cycle<n:
            for i in prange(n):
                self.step()
                cycle+=1
                
        while cycle<2*n and cycle>=n:
            self.simulation_type == 'RestrictiveMonetaryPolicy'
            for i in prange(n):
                self.step()
                cycle+=1
                
        self.running = False        
        
    def normalize_banks(self):
        # Normalize banks size and Compute market share (in % of total assets)
        total_size = sum([_.initialSize for _ in self.schedule.banks])
        factor = self.numberBanks / total_size
        for bank in self.schedule.banks:
            bank.marketShare = bank.initialSize / total_size
            bank.initialSize *= factor
   

    @staticmethod
    def update_exogeneous_factors(exogenous_factors, number_of_banks):
        if isinstance(exogenous_factors, dict):
            for key, value in exogenous_factors.items():
                setattr(ExogenousFactors, key, value)

        if number_of_banks:
            ExogenousFactors.numberBanks = number_of_banks

    @staticmethod
    def update_exogeneous_factors_by_simulation_type(simulation_type):
        if simulation_type == SimulationType.HighSpread:
            pass
        if simulation_type == SimulationType.LowSpread:
            ExogenousFactors.standardCorporateClientLoanInterestRate = 0.06
        elif simulation_type == SimulationType.ClearingHouse:
            ExogenousFactors.isClearingGuaranteeAvailable = True
        elif simulation_type == SimulationType.ClearingHouseLowSpread:
            ExogenousFactors.isClearingGuaranteeAvailable = True
            ExogenousFactors.standardCorporateClientLoanInterestRate = 0.06
        elif simulation_type == SimulationType.Basel:
            ExogenousFactors.standardCorporateClients = False
            ExogenousFactors.isCentralBankZeroIntelligenceAgent = False
            ExogenousFactors.isCapitalRequirementActive = True
            ExogenousFactors.interbankPriority = InterbankPriority.RiskSorted
            ExogenousFactors.standardCorporateClientDefaultRate = 0.05
        elif simulation_type == SimulationType.BaselBenchmark:
            ExogenousFactors.standardCorporateClients = False
            ExogenousFactors.standardCorporateClientDefaultRate = 0.05
        elif simulation_type == SimulationType.DepositInsurance:
            ExogenousFactors.areDepositorsZeroIntelligenceAgents = False
            ExogenousFactors.isDepositInsuranceAvailable = True
        elif simulation_type == SimulationType.DepositInsuranceBenchmark:
            ExogenousFactors.areDepositorsZeroIntelligenceAgents = False

        elif simulation_type == SimulationType.RestrictiveMonetaryPolicy:
            ExogenousFactors.interbankInterestRate = 0.02
            ExogenousFactors.LowRiskCorporateClientDefaultRate = 0.06
            ExogenousFactors.HighRiskCorporateClientDefaultRate = 0.10
            ExogenousFactors.HighRiskCorporateClientLoanInterestRate = 0.10
            ExogenousFactors.LowRiskCorporateClientLoanInterestRate = 0.08
            ExogenousFactors.probabilityofWithdrawal = 0.20
        elif simulation_type == SimulationType.ExpansiveMonetaryPolicy:
            ExogenousFactors.interbankInterestRate = 0.005
            ExogenousFactors.LowRiskCorporateClientDefaultRate = 0.02
            ExogenousFactors.HighRiskCorporateClientDefaultRate = 0.03
            ExogenousFactors.HighRiskCorporateClientLoanInterestRate = 0.12
            ExogenousFactors.LowRiskCorporateClientLoanInterestRate = 0.045
            ExogenousFactors.probabilityofWithdrawal = 0.10
