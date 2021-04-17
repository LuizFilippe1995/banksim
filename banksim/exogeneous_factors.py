from enum import Enum


class SimulationType(Enum):
    HighSpread = 1
    LowSpread = 2
    ClearingHouse = 3
    ClearingHouseLowSpread = 4
    Basel = 5
    BaselBenchmark = 6
    DepositInsurance = 7
    DepositInsuranceBenchmark = 8
    RestrictiveMonetaryPolicy = 9
    ExpansiveMonetaryPolicy = 10

    
class BankSizeDistribution(Enum):
    Vanilla = 1
    LogNormal = 2


class InterbankPriority(Enum):
    Random = 1
    RiskSorted = 2


class ExogenousFactors:
    # Model
    numberBanks = 50
    depositInterestRate = 0.005 
    interbankInterestRate = 0.01  
    liquidAssetsInterestRate = 0
    illiquidAssetDiscountRate = 0.15
    interbankLendingMarketAvailable = True
    banksMaySellNonLiquidAssetsAtDiscountPrices = True
    banksHaveLimitedLiability = False

    # Banks
    bankSizeDistribution = BankSizeDistribution.Vanilla
    numberDepositorsPerBank = 100
    numberCorporateClientsPerBank = 50
    areBanksZeroIntelligenceAgents = False

    # Central Bank
    centralBankLendingInterestRate = 0.04 
    offersDiscountWindowLending = True
    minimumCapitalAdequacyRatio = -10
    isCentralBankZeroIntelligenceAgent = True
    isCapitalRequirementActive = True # era false
    isTooBigToFailPolicyActive = False
    isDepositInsuranceAvailable = True # era false

    # Clearing House
    isClearingGuaranteeAvailable = True
    interbankPriority = InterbankPriority.Random

    # Depositors
    areDepositorsZeroIntelligenceAgents = True
    areBankRunsPossible = True
    amountWithdrawn = 1.0
    probabilityofWithdrawal = 0.15

    # Firms / Corporate Clients
    HighRiskCorporateClientDefaultRate = 0.07
    HighRiskCorporateClientLossGivenDefault = 1
    HighRiskCorporateClientLoanInterestRate = 0.08
    LowRiskCorporateClientDefaultRate = 0.04
    LowRiskCorporateClientLoanInterestRate = 0.06
    LowRiskCorporateClientLossGivenDefault = 1
    
    # Risk Weights
    CashRiskWeight = 0
    HighRiskCorporateLoanRiskWeight = 1
    InterbankLoanRiskWeight = 1
    LowRiskCorporateLoanRiskWeight = 0.8
    
    # Learning
    DefaultEWADampingFactor = 1
