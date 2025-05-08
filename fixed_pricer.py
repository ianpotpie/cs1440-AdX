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

DEFAULT_CAMPAIGN_VALUES = {
        MarketSegment({'Young', 'Male'}): 148.8045156661962,
 MarketSegment({'Old', 'Male'}): 127.71000298272882,
 MarketSegment({'Male', 'LowIncome'}): 11.173146064417898,
 MarketSegment({'HighIncome', 'Male'}): 221.97075219272625,
 MarketSegment({'Young', 'Female'}): 34.40009861326649,
 MarketSegment({'Old', 'Female'}): 82.05431995947367,
 MarketSegment({'Female', 'LowIncome'}): 17.06266101035064,
 MarketSegment({'Female', 'HighIncome'}): 210.4621089044885,
 MarketSegment({'Young', 'LowIncome'}): 36.0392526387527,
 MarketSegment({'Young', 'HighIncome'}): 121.87737747813175,
 MarketSegment({'Old', 'LowIncome'}): 121.82813261714642,
 MarketSegment({'Old', 'HighIncome'}): 15.471433243045018,
 MarketSegment({'Young', 'Male', 'LowIncome'}): 30.932644858075832,
 MarketSegment({'Young', 'Male', 'HighIncome'}): 127.54192425729411,
 MarketSegment({'Old', 'Male', 'LowIncome'}): 200.75369449956352,
 MarketSegment({'Old', 'HighIncome', 'Male'}): 44.94749178024689,
 MarketSegment({'Young', 'Female', 'LowIncome'}): 276.12646354944195,
 MarketSegment({'Young', 'Female', 'HighIncome'}): 13.344842795877803,
 MarketSegment({'Old', 'Female', 'LowIncome'}): 144.81581233901997,
 MarketSegment({'Old', 'Female', 'HighIncome'}): 141.84994734090145}

DEFAULT_AD_VALUES = {MarketSegment({'Young', 'Male'}): {MarketSegment({'Young', 'Male'}): 0.6173006225446774},
 MarketSegment({'Old', 'Male'}): {MarketSegment({'Old', 'Male'}): 1.4483893345527383},
 MarketSegment({'Male', 'LowIncome'}): {MarketSegment({'Male', 'LowIncome'}): 90.15621858602097},
 MarketSegment({'HighIncome', 'Male'}): {MarketSegment({'HighIncome', 'Male'}): 7.441951442876128},
 MarketSegment({'Young', 'Female'}): {MarketSegment({'Young', 'Female'}): 0.47576954608099814},
 MarketSegment({'Old', 'Female'}): {MarketSegment({'Old', 'Female'}): 0.9503920429158154},
 MarketSegment({'Female', 'LowIncome'}): {MarketSegment({'Female', 'LowIncome'}): 25.746282003408666},
 MarketSegment({'Female', 'HighIncome'}): {MarketSegment({'Female', 'HighIncome'}): 31.716049564779368},
 MarketSegment({'Young', 'LowIncome'}): {MarketSegment({'Young', 'LowIncome'}): 33.91974237727237},
 MarketSegment({'Young', 'HighIncome'}): {MarketSegment({'Young', 'HighIncome'}): 65.17285678946114},
 MarketSegment({'Old', 'LowIncome'}): {MarketSegment({'Old', 'LowIncome'}): 0.806048298798337},
 MarketSegment({'Old', 'HighIncome'}): {MarketSegment({'Old', 'HighIncome'}): 57.90154163042069},
 MarketSegment({'Young', 'Male', 'LowIncome'}): {MarketSegment({'Young', 'Male'}): 10.461257763327167,
                                                 MarketSegment({'Male', 'LowIncome'}): 40.38258045593593,
                                                 MarketSegment({'Young', 'LowIncome'}): 100.45308564518375,
                                                 MarketSegment({'Young', 'Male', 'LowIncome'}): 2.181921592680982},
 MarketSegment({'Young', 'Male', 'HighIncome'}): {MarketSegment({'Young', 'Male'}): 8.835697637322724,
                                                  MarketSegment({'HighIncome', 'Male'}): 24.02377240474992,
                                                  MarketSegment({'Young', 'HighIncome'}): 8.378665773004098,
                                                  MarketSegment({'Young', 'Male', 'HighIncome'}): 5.172331836066506},
 MarketSegment({'Old', 'Male', 'LowIncome'}): {MarketSegment({'Old', 'Male'}): 97.9520534766225,
                                               MarketSegment({'Male', 'LowIncome'}): 105.2457278577288,
                                               MarketSegment({'Old', 'LowIncome'}): 70.86424469740541,
                                               MarketSegment({'Old', 'Male', 'LowIncome'}): 38.08057165093329},
 MarketSegment({'Old', 'HighIncome', 'Male'}): {MarketSegment({'Old', 'Male'}): 44.280538508729705,
                                                MarketSegment({'HighIncome', 'Male'}): 97.740830070494,
                                                MarketSegment({'Old', 'HighIncome'}): 32.282844724851614,
                                                MarketSegment({'Old', 'HighIncome', 'Male'}): 40.2238630693996},
 MarketSegment({'Young', 'Female', 'LowIncome'}): {MarketSegment({'Young', 'Female'}): 18.290287375045995,
                                                   MarketSegment({'Female', 'LowIncome'}): 90.64690735360674,
                                                   MarketSegment({'Young', 'LowIncome'}): 39.67694285969217,
                                                   MarketSegment({'Young', 'Female', 'LowIncome'}): 7.197005947173487},
 MarketSegment({'Young', 'Female', 'HighIncome'}): {MarketSegment({'Young', 'Female'}): 29.349209483329815,
                                                    MarketSegment({'Female', 'HighIncome'}): 94.83786224109008,
                                                    MarketSegment({'Young', 'HighIncome'}): 39.95891953585195,
                                                    MarketSegment({'Young', 'Female', 'HighIncome'}): 10.317755333739496},
 MarketSegment({'Old', 'Female', 'LowIncome'}): {MarketSegment({'Old', 'Female'}): 47.38202721597156,
                                                 MarketSegment({'Female', 'LowIncome'}): 85.33785856150035,
                                                 MarketSegment({'Old', 'LowIncome'}): 92.00638765554999,
                                                 MarketSegment({'Old', 'Female', 'LowIncome'}): 92.35595228602807},
 MarketSegment({'Old', 'Female', 'HighIncome'}): {MarketSegment({'Old', 'Female'}): 22.209827331649123,
                                                  MarketSegment({'Female', 'HighIncome'}): 4.083493389945208,
                                                  MarketSegment({'Old', 'HighIncome'}): 30.708358033206164,
                                                  MarketSegment({'Old', 'Female', 'HighIncome'}): 44.82329773433364}}
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
