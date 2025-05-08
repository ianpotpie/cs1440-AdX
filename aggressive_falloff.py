from typing import Dict, Set

from adx.adx_game_simulator import AdXGameSimulator
from adx.agents import NDaysNCampaignsAgent
from adx.structures import Bid, BidBundle, Campaign
from adx.tier1_ndays_ncampaign_agent import Tier1NDaysNCampaignsAgent


class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self):
        # TODO: fill this in (if necessary)
        super().__init__()
        self.name = "Aggressive Bot"  # TODO: enter a name.

    def on_new_game(self) -> None:
        # TODO: fill this in (if necessary)
        print()

    def get_ad_bids(self) -> Set[BidBundle]:
        bundles = set()

        for campaign in self.campaigns:
            

        return bundles

    def get_campaign_bids(
        self, campaigns_for_auction: Set[Campaign]
    ) -> Dict[Campaign, float]:
        bids = {}

        for campaign in campaigns_for_auction:
            bids[campaign] = self.clip_campaign_bid(campaign, 0.0)

        return bids


if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [MyNDaysNCampaignsAgent()] + [
        Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)
    ]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=500)
