class BankEWAStrategy:
    # capital ratio (capital / assets)
    numberAlphaOptions = 30
    # liquidity ratio(liquid assets / deposits)
    numberBetaOptions = 20
    # risk appetite (High Risk Corporate Client/ Low Risk Corporate Client)
    numberGammaOptions = 30

    def __init__(self, alpha_index_option=0, beta_index_option=0, gamma_index_option=0):
        self.alphaIndex = alpha_index_option
        self.betaIndex = beta_index_option
        self.gammaIndex = gamma_index_option
        self.strategyProfit = self.strategyProfitPercentage = self.strategyProfitPercentageDamped = 0
        self.A = self.P = self.F = 0
    
    
    def get_alpha_value(self):
        return (self.alphaIndex + 1) / 100 
    
    def get_beta_value(self):
        return  (self.betaIndex + 1) / 100 
    
    def get_gamma_value(self):
        return (self.gammaIndex + 1) / 100
        
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.alphaIndex == other.alphaIndex and self.betaIndex == other.betaIndex and \
        self.gammaIndex == other.gammaIndex
        return False
    
    def reset(self):
        self.strategyProfit = self.strategyProfitPercentage = self.strategyProfitPercentageDamped = 0
        self.A = self.P = self.F = 0

    @classmethod
    def bank_ewa_strategy_list(cls):
        return [BankEWAStrategy(a, b, c) for a in range(cls.numberAlphaOptions) for b in \
                range(cls.numberBetaOptions) for c in range(cls.numberGammaOptions)]

                
