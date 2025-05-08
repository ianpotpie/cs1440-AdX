import gc
from typing import Dict, Set

from adx.adx_game_simulator import AdXGameSimulator
from adx.agents import NDaysNCampaignsAgent
from adx.structures import Bid, BidBundle, Campaign
from adx.tier1_ndays_ncampaign_agent import Tier1NDaysNCampaignsAgent


class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self):
        # TODO: fill this in (if necessary)
        super().__init__()
        self.name = "Common Sense Agent"  # TODO: enter a name.

    def on_new_game(self) -> None:
        # TODO: fill this in (if necessary)
        self.simulator = None
        for referrer in gc.get_referrers(self):
            if isinstance(referrer, AdXGameSimulator):
                self.simulator = referrer
                break
            for meta_referrer in gc.get_referrers(referrer):
                if isinstance(meta_referrer, AdXGameSimulator):
                    self.simulator = meta_referrer
                    break
        input()

    def get_ad_bids(self) -> Set[BidBundle]:
        # TODO: fill this in
        bundles = set()

        return bundles

    def get_campaign_bids(
        self, campaigns_for_auction: Set[Campaign]
    ) -> Dict[Campaign, float]:
        # TODO: fill this in
        bids = {}

        return bids


if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [MyNDaysNCampaignsAgent()] + [
        Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)
    ]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=500)
