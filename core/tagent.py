import logging

class TradingAgent():
    """
    This agent class is made to interact with a trading gym we made in here #TODO : add link that
    """
    def __init__(self, buy_commission=0.015, hold_commission=0.315, start_money=1000000, max_count_buyhold=100, ):
        self.buy_commission = 0.015
        self.hold_commission = 0.315
        self.start_money = 1000000
        self.max_count_buy_or_hold = 100

    def _is_done_from_agent(self):
        """
        check whether or not this current episode is finished based on its agent status
        This will be called right after checking done variable from the trading gym to double check

        Environment side
        ========================
        . current step exceeds in max_steps

        Agent side
        ========================
        . lose all money agent has (= no balance)
        . exceed the count of available transaction

        :return:
            True or False
        """
        pass

    def _get_status(self):
        """
        agent status information
        for now, we can only allow a agent to have only one ticker when its agent trades.

        :parameter
        . balance
        . num_equitiy_to_hold
        . remaining_count_to_buyhold
        .
        :return:
        """
        pass



    def _transform_obersvation