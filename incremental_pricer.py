import itertools
import os.path
import pickle
import pprint
import random
import statistics
from copy import deepcopy
from typing import Dict, Set

import numpy as np

from adx.adx_game_simulator import AdXGameSimulator
from adx.agents import NDaysNCampaignsAgent
from adx.structures import Bid, BidBundle, Campaign, MarketSegment
from adx.tier1_ndays_ncampaign_agent import Tier1NDaysNCampaignsAgent

# campaign bid range is [0.1R, R]
# the campaign reach distribution is [0.3,0.5,0.7] (multiplied by segment pop to get reach)
# the lowest segment pop is MarketSegment(("Female", "Young", "HighIncome")): 256,
# the highest segment pop is MarketSegment(("Female", "LowIncome")): 4381,
# so reaches range from 0.3 * 256 = 76.8 to 3066.7
# overall campaign bids can range from 7.68 to 3066.7
DEFAULT_CAMPAIGN_VALUES = {segment: 10.0 for segment in MarketSegment.all_segments()}

DEFAULT_CAMPAIGN_LIMITS = {segment: 10.0 for segment in MarketSegment.all_segments()}

DEFAULT_AD_VALUES = {
    segment: {
        ad_target: 0.0
        for ad_target in MarketSegment.all_segments()
        if ad_target <= segment
    }
    for segment in MarketSegment.all_segments()
}

DEFAULT_AD_LIMITS = {
    segment: {
        ad_target: 0.0
        for ad_target in MarketSegment.all_segments()
        if ad_target <= segment
    }
    for segment in MarketSegment.all_segments()
}

bot_id = itertools.count(1)


class FixedPricingAgent(NDaysNCampaignsAgent):

    def __init__(
        self,
        campaign_values,
        campaign_limits,
        ad_values,
        ad_limits,
        name="The Price is Right",
    ):
        # TODO: fill this in (if necessary)
        super().__init__()
        self.name = name
        self.campaign_values = campaign_values
        self.campaign_limits = campaign_limits
        self.ad_values = ad_values
        self.ad_limits = ad_limits

    def on_new_game(self) -> None:
        # TODO: fill this in (if necessary)
        pass

    def get_ad_bids(self) -> Set[BidBundle]:
        bundles = set()

        for campaign in self.get_active_campaigns():
            bids = set()
            ad_values = self.ad_values[campaign.target_segment]
            ad_limits = self.ad_limits[campaign.target_segment]
            for ad, price in ad_values.items():
                bid_price = max(price, 0)
                bid_limit = max(price + 1e-8, ad_limits[ad])
                bids.add(Bid(self, ad, bid_price, bid_limit))
            # TODO: think about appropriate spending limit
            uid = campaign.uid
            campaign_limit = self.campaign_limits[campaign.target_segment]
            bundles.add(BidBundle(uid, campaign_limit, bids))

        return bundles

    def get_campaign_bids(
        self, campaigns_for_auction: Set[Campaign]
    ) -> Dict[Campaign, float]:
        # campaign bids must fall in the range [0.1 * Reach, Reach]
        # i.e. between 0.1 and 1.0 per person
        bids = {}

        for campaign in campaigns_for_auction:
            bids[campaign] = self.clip_campaign_bid(
                campaign, self.campaign_values[campaign.target_segment] / campaign.reach
            )

        return bids


def reproduce_bots(old_bots, n, baseline_std=0.1):
    new_bots = []
    for _ in range(n):
        new_bot = FixedPricingAgent(
            name=f"TPiR ({next(bot_id)})",
            campaign_values=deepcopy(old_bots[0].campaign_values),
            campaign_limits=deepcopy(old_bots[0].campaign_limits),
            ad_values=deepcopy(old_bots[0].ad_values),
            ad_limits=deepcopy(old_bots[0].ad_limits),
        )
        new_bots.append(new_bot)

    for segment in old_bots[0].campaign_values:
        old_vals = list(map(lambda x: x.campaign_values[segment], old_bots))
        segment_mean = statistics.mean(old_vals)
        segment_std = statistics.stdev(old_vals) + baseline_std
        for bot in new_bots:
            bot.campaign_values[segment] = min(
                3067, max(7, np.random.normal(segment_mean, segment_std))
            )

    for segment in old_bots[0].campaign_limits:
        old_vals = list(map(lambda x: x.campaign_limits[segment], old_bots))
        segment_mean = statistics.mean(old_vals)
        segment_std = statistics.stdev(old_vals) + baseline_std
        for bot in new_bots:
            bot.campaign_limits[segment] = abs(
                np.random.normal(segment_mean, segment_std)
            )

    for segment in old_bots[0].ad_values:
        for ad in old_bots[0].ad_values[segment]:
            old_vals = list(map(lambda x: x.ad_values[segment][ad], old_bots))
            segment_mean = statistics.mean(old_vals)
            segment_std = statistics.stdev(old_vals) + baseline_std
            for bot in new_bots:
                bot.ad_values[segment][ad] = abs(
                    np.random.normal(segment_mean, segment_std)
                )

    for segment in old_bots[0].ad_limits:
        for ad in old_bots[0].ad_limits[segment]:
            old_vals = list(map(lambda x: x.ad_limits[segment][ad], old_bots))
            segment_mean = statistics.mean(old_vals)
            segment_std = statistics.stdev(old_vals) + baseline_std
            for bot in new_bots:
                bot.ad_limits[segment][ad] = abs(
                    np.random.normal(segment_mean, segment_std)
                )

    return new_bots


if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    # test_agents = [MyNDaysNCampaignsAgent()] + [
    #     Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)
    # ]

    test_agents = [
        FixedPricingAgent(
            DEFAULT_CAMPAIGN_VALUES,
            DEFAULT_CAMPAIGN_LIMITS,
            DEFAULT_AD_VALUES,
            DEFAULT_AD_LIMITS,
            name=f"TPiR ({next(bot_id)})",
        )
        for _ in range(2)
    ]
    test_agents = reproduce_bots(test_agents, 20, baseline_std=1)
    # Don't change this. Adapt initialization to your environment
    n_epochs = 100_000
    simulator = AdXGameSimulator()
    for i in range(n_epochs):
        profits = simulator.run_simulation(agents=test_agents, num_simulations=10)
        print(sum(profits.values()) / len(profits))

        # pick out the best agents
        best_agents = sorted(
            test_agents, key=lambda x: x.get_cumulative_profit(), reverse=True
        )[:10]

        test_agents = reproduce_bots(best_agents, 10) + best_agents

        # save the prices/limits to pickle files
        if i % 100 == 0:
            with open("fixed_pricer/best_bots.pkl", "wb") as f:
                vals = {
                    "campaign_values": best_agents[0].campaign_values,
                    "campaign_limits": best_agents[0].campaign_limits,
                    "ad_values": best_agents[0].ad_values,
                    "ad_limits": best_agents[0].ad_limits,
                }
                pickle.dump(vals, f)

            pprint.pprint(test_agents[0].campaign_values)
            pprint.pprint(test_agents[0].ad_values)

    # print(simulator.states)
    # print(vars(simulator))
