import itertools
from typing import Dict, Set

import numpy as np
from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.agents.utils.adx.structures import (
    Bid,
    BidBundle,
    Campaign,
    MarketSegment,
)
from agt_server.local_games.adx_arena import AdXGameSimulator

segment_pop = {
    MarketSegment(("Male", "Young")): 2353,
    MarketSegment(("Male", "Old")): 2603,
    MarketSegment(("Male", "LowIncome")): 3631,
    MarketSegment(("Male", "HighIncome")): 1325,
    MarketSegment(("Female", "Young")): 2236,
    MarketSegment(("Female", "Old")): 2808,
    MarketSegment(("Female", "LowIncome")): 4381,
    MarketSegment(("Female", "HighIncome")): 663,
    MarketSegment(("Young", "LowIncome")): 3816,
    MarketSegment(("Young", "HighIncome")): 773,
    MarketSegment(("Old", "LowIncome")): 4196,
    MarketSegment(("Old", "HighIncome")): 1215,
    MarketSegment(("Male", "Young", "LowIncome")): 1836,
    MarketSegment(("Male", "Young", "HighIncome")): 517,
    MarketSegment(("Male", "Old", "LowIncome")): 1795,
    MarketSegment(("Male", "Old", "HighIncome")): 808,
    MarketSegment(("Female", "Young", "LowIncome")): 1980,
    MarketSegment(("Female", "Young", "HighIncome")): 256,
    MarketSegment(("Female", "Old", "LowIncome")): 2401,
    MarketSegment(("Female", "Old", "HighIncome")): 407,
}


def calculate_effective_reach(x: int | np.ndarray, R: int) -> float:
    return (2.0 / 4.08577) * (
        np.atan(4.08577 * ((x + 0.0) / R) - 3.08577) - np.atan(-3.08577)
    )


class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self):
        super().__init__()
        self.name = "Basic Bot"
        self.aggression = 0.0
        self.ad_margin = 0.0

        self.initial_params = itertools.product(
            np.linspace(0.0, 1.0, 10),  # range of aggressions
            np.linspace(0.0, 0.5, 10),  # range of margin values
        )

        self.prev_agressions = []
        self.prev_ad_margins = []
        self.prev_profits = []

    def on_new_game(self) -> None:
        print()
        params = next(self.initial_params, None)
        if params is not None:
            self.aggression, self.ad_margin = params
            return

        self.aggression = max(
            0, min(1, np.average(self.prev_agressions, weights=self.prev_profits))
        )
        self.ad_margin = max(
            0, min(1, np.average(self.prev_ad_margins, weights=self.prev_profits))
        )

    def get_ad_bids(self) -> Set[BidBundle]:
        # TODO: fill this in
        bundles = set()
        for campaign in self.get_active_campaigns():
            bids = set()

            curr_reach = self.get_cumulative_reach(campaign)
            target_segment = campaign.target_segment
            target_size = segment_pop[target_segment]
            max_reach = min(curr_reach + target_size, campaign.reach * 2)
            next_reaches = np.linspace(curr_reach + 1, max_reach, 500)
            curr_effective_reach = calculate_effective_reach(curr_reach, campaign.reach)
            next_effective_reaches = calculate_effective_reach(
                next_reaches, campaign.reach
            )
            unit_revenues = (
                (next_effective_reaches - curr_effective_reach)
                * campaign.budget
                / (next_reaches - curr_reach)
            )

            best_idx = np.argmax(unit_revenues)
            best_unit_revenue = unit_revenues[best_idx]
            best_next_reach = next_reaches[best_idx]
            best_unit_count = best_next_reach - curr_reach

            bid_per_item = best_unit_revenue * (1 - self.ad_margin)
            bid_limit = best_unit_count * bid_per_item
            if bid_limit > 0:
                bid = Bid(self, target_segment, bid_per_item, bid_limit)
                bids.add(bid)
            bundle = BidBundle(
                campaign_id=campaign.uid, limit=bid_limit, bid_entries=bids
            )
            bundles.add(bundle)

        return bundles

    def get_campaign_bids(
        self, campaigns_for_auction: Set[Campaign]
    ) -> Dict[Campaign, float]:
        bids = {}

        for campaign in campaigns_for_auction:
            bid = campaign.reach * (1 - 0.9 * self.aggression)
            bids[campaign] = self.clip_campaign_bid(campaign, bid)

        # save the profits here since the campaign auction happens at the end of the day
        # only save on the last day
        if self.current_day == 1:
            self.prev_profits.append(self.profit)
            self.prev_agressions.append(self.aggression)
            self.prev_ad_margins.append(self.ad_margin)
        else:
            self.prev_profits[-1] = self.profit

        return bids


agent_submission = MyNDaysNCampaignsAgent()
