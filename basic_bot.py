import itertools
import random
from math import atan
from typing import Dict, Set

import numpy as np

from adx.adx_game_simulator import AdXGameSimulator
from adx.agents import NDaysNCampaignsAgent
from adx.structures import Bid, BidBundle, Campaign, MarketSegment
from adx.tier1_ndays_ncampaign_agent import Tier1NDaysNCampaignsAgent

# todays_profit += (
#     effective_reach
# ) * agent_state.budgets[campaign.uid] - total_cost

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
        # TODO: fill this in (if necessary)
        super().__init__()
        self.name = "Basic Bot"  # TODO: enter a name.
        self.aggression = 0.0
        self.ad_margin = 0.1

        # self.initial_params = itertools.product(
        #     np.linspace(0.0, 1.0, 5),  # range of aggressions
        #     np.linspace(0.0, 0.5, 5),  # range of margin values
        # )

        self.prev_agressions = []
        self.prev_ad_margins = []
        self.prev_profits = []

    def on_new_game(self) -> None:
        if len(self.prev_profits) == 0:
            return
        # params = next(self.initial_params, None)
        # if params is not None:
        #     self.aggression, self.ad_margin = params
        #     return

        print(
            f"Prev profit was {self.prev_profits[-1]:.2f} with aggression {self.aggression:.3f} and ad margin {self.ad_margin:.3f}"
        )

        best_round = np.argmax(self.prev_profits[-50:])

        if self.prev_profits[best_round] < 1500:
            self.aggression = random.random()
            self.ad_margin = random.random()
            return

        # best_aggression = self.prev_agressions[-50:][best_round]
        # best_ad_margin = self.prev_ad_margins[-50:][best_round]

        best_aggression = np.average(
            self.prev_agressions[-50:], weights=self.prev_profits[-50:]
        )
        best_ad_margin = np.average(
            self.prev_ad_margins[-50:], weights=self.prev_profits[-50:]
        )

        self.aggression = max(
            0, min(1, best_aggression + np.random.uniform(-0.05, 0.05))
        )
        self.ad_margin = max(0, min(1, best_ad_margin + np.random.uniform(-0.05, 0.05)))
        # self.aggression = max(
        #     0, min(1, np.average(self.prev_agressions, weights=self.prev_profits))
        # )
        # self.ad_margin = max(
        #     0, min(1, np.average(self.prev_ad_margins, weights=self.prev_profits))
        # )

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


if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [MyNDaysNCampaignsAgent()] + [
        Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)
    ]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=500)
