import random
from typing import List, Dict, Optional
from adx.structures import Campaign, Bid, BidBundle, MarketSegment
from adx.pmfs import PMF
from adx.tier1_ndays_ncampaign_agent import Tier1NDaysNCampaignsAgent
from adx.states import CampaignBidderState
from collections import defaultdict
from adx.agents import NDaysNCampaignsAgent
from math import isfinite, atan

CONFIG = {
        'num_agents': 10,
        'num_days': 10,
        'quality_score_alpha': 0.5,
        'campaigns_per_day': 5,
        'campaign_reach_dist': [0.3,0.5,0.7],
        'campaign_length_dist': [1, 2, 3],
        'market_segment_dist': [
            MarketSegment(("Male", "Young")),
            MarketSegment(("Male", "Old")),
            MarketSegment(("Male", "LowIncome")),
            MarketSegment(("Male", "HighIncome")),
            MarketSegment(("Female", "Young")),
            MarketSegment(("Female", "Old")),
            MarketSegment(("Female", "LowIncome")),
            MarketSegment(("Female", "HighIncome")),
            MarketSegment(("Young", "LowIncome")),
            MarketSegment(("Young", "HighIncome")),
            MarketSegment(("Old", "LowIncome")),
            MarketSegment(("Old", "HighIncome")),
            MarketSegment(("Male", "Young", "LowIncome")),
            MarketSegment(("Male", "Young", "HighIncome")),
            MarketSegment(("Male", "Old", "LowIncome")),
            MarketSegment(("Male", "Old", "HighIncome")),
            MarketSegment(("Female", "Young", "LowIncome")),
            MarketSegment(("Female", "Young", "HighIncome")),
            MarketSegment(("Female", "Old", "LowIncome")),
            MarketSegment(("Female", "Old", "HighIncome"))
        ],
        'market_segment_pop': {
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
            MarketSegment(("Female", "Old", "HighIncome")): 407
        },
        'user_segment_pmf': {
            MarketSegment(("Male", "Young", "LowIncome")): 0.1836,
            MarketSegment(("Male", "Young", "HighIncome")): 0.0517,
            MarketSegment(("Male", "Old", "LowIncome")): 0.1795,
            MarketSegment(("Male", "Old", "HighIncome")): 0.0808,
            MarketSegment(("Female", "Young", "LowIncome")): 0.1980,
            MarketSegment(("Female", "Young", "HighIncome")): 0.0256,
            MarketSegment(("Female", "Old", "LowIncome")): 0.2401,
            MarketSegment(("Female", "Old", "HighIncome")): 0.0407
        }
    }

def calculate_effective_reach(x: int, R: int) -> float:
    return (2.0 / 4.08577) * (atan(4.08577 * ((x + 0.0) / R) - 3.08577) - atan(-3.08577))


class AdXGameSimulator:
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = CONFIG
        self.num_agents = config['num_agents']
        self.num_days = config['num_days']
        self.α = config['quality_score_alpha']
        self.campaigns_per_day = config['campaigns_per_day']
        self.agents: List = []
        self.campaign_reach_dist = config['campaign_reach_dist']
        self.campaign_length_dist = config['campaign_length_dist']
        self.market_segment_dist = config['market_segment_dist']
        self.market_segment_pop = config['market_segment_pop']
        self.user_segment_dist = PMF(config['user_segment_pmf'])
        self.sub_segments = defaultdict(list)
        for user_seg in config['user_segment_pmf']:
            for market_seg in config['market_segment_dist']:
                if market_seg.issubset(user_seg):
                    self.sub_segments[user_seg].append(market_seg)

    def init_agents(self, agent_types: List[type]) -> Dict[NDaysNCampaignsAgent, CampaignBidderState]:
        states = dict()
        self.agents = []
        for i, agent_type in enumerate(agent_types):
            agent_type.init()
            self.agents.append(agent_type)
            self.agents[i].agent_num = i
            states[agent_type] = CampaignBidderState(i)
        return states 
    
    def generate_campaign(self, start_day: int, end_day: Optional[int] = None) -> Campaign:
        delta = random.choice(self.campaign_reach_dist)
        length =  random.choice(self.campaign_length_dist)
        mkt_segment = random.choice(self.market_segment_dist)
        reach = int(self.market_segment_pop[mkt_segment] * delta)
        if end_day is None:
            end_day=start_day + length - 1
        return Campaign(reach=reach, 
                        target=mkt_segment, 
                        start_day=start_day, 
                        end_day=end_day)
    
    def is_valid_campaign_bid(self, bid: float, reach: int) -> bool:
        return isfinite(bid) and 0.1 * reach <= bid <= reach
        
    def is_valid_bid(self, bid: Bid) -> bool:
        bid = bid.bid_per_item
        return isfinite(bid) and bid > 0

    def run_ad_auctions(self, bid_bundles : List[BidBundle], users: List[MarketSegment], day: int) -> None:
        bidder_states = self.states
        # Map for market_segment to bid
        seg_to_bid = defaultdict(set)
        # Map for campaign_id to limit 
        daily_limits = dict() 
        # Map for bid to bundle 
        bid_to_bundle = dict() 
        # Map from bid entry to spend 
        bid_to_spend = dict()
        for bid_bundle in bid_bundles:
            daily_limits[bid_bundle.campaign_id] = bid_bundle.limit 
            # If campaign does not exist throw-away the bid bundle
            if bid_bundle.campaign_id not in self.campaigns: continue 
            # If campaign is not active throw-away the bid bundle
            campaign = self.campaigns[bid_bundle.campaign_id]
            if not (campaign.start_day <= day and campaign.end_day >= day): continue
            
            for bid in bid_bundle.bid_entries:
                if self.is_valid_bid(bid):
                    bid_to_bundle[bid] = bid_bundle 
                    seg_to_bid[bid.item].add(bid)
                    bid_to_spend[bid] = 0

        for user_segment in users:
            # Get bids that match user
            bids = []
            sub_segments = self.sub_segments[user_segment]
            for seg in sub_segments:
                bids.extend(seg_to_bid[seg])
            
            bids.sort(key=lambda b: b.bid_per_item, reverse=True)

            for i, bid in enumerate(bids):
                campaign_id = bid_to_bundle[bid].campaign_id
                price = bids[i + 1].bid_per_item if i + 1 < len(bids) else 0     
                bidder_states[bid.bidder].spend[campaign_id] += price
                over_bid_limit = bid_to_spend[bid] + price > bid.bid_limit
                over_bundle_limit = bidder_states[bid.bidder].spend[campaign_id] + price > daily_limits[campaign_id]
                if over_bid_limit or over_bundle_limit:
                    # Remove bid if over limit
                    seg_to_bid[bid.item].remove(bid)
                    continue 
                else: 
                    # Update bid 
                    bid_to_spend[bid] += price
                    bidder_state = bidder_states[bid.bidder]
                    campaign_id = bid_to_bundle[bid].campaign_id
                    campaign = bidder_state.campaigns[campaign_id]
                    if campaign is not None:
                        bidder_state.spend[campaign_id] += price
                        campaign.cumulative_cost += price
                        if campaign.target_segment.issubset(user_segment):
                            bidder_state.impressions[campaign_id] += 1
                            campaign.cumulative_reach += 1 
                    break 


    def run_campaign_auctions(self, agent_bids: Dict[NDaysNCampaignsAgent, Dict[Campaign, float]], new_campaigns: List[Campaign]) -> None:
        new_campaigns = set(new_campaigns)
        for campaign in new_campaigns:
            bids = []
            for agent in self.agents:
                if campaign in agent_bids[agent]:
                    reach = campaign.reach
                    agent_bid = agent_bids[agent][campaign]
                    if self.states[agent].quality_score > 0 and self.is_valid_campaign_bid(agent_bid, reach):
                        effective_bid = agent_bid / self.states[agent].quality_score
                        bids.append((agent, effective_bid))
                   
            if bids:
                winner, effective_bid = min(bids, key=lambda x: x[1])
                if len(bids) == 1:
                    q_low = 0.0
                    all_quality_scores = [self.states[agent].quality_score for agent in self.agents]
                    sorted_quality_scores = sorted(all_quality_scores)
                    for i in range(3):
                        q_low += sorted_quality_scores[min(i, len(sorted_quality_scores) - 1)]
                    q_low /= 3
                    budget = campaign.reach / q_low * self.states[winner].quality_score
                else:
                    second_lowest_bid = sorted(bids, key=lambda x: x[1])[1][1]
                    budget = second_lowest_bid * self.states[winner].quality_score
                campaign.budget = budget
                winner.my_campaigns.add(campaign)
                winner_state = self.states[winner]
                winner_state.add_campaign(campaign)
                self.campaigns[campaign.uid] = campaign

    def generate_auction_items(self, num_items: int) -> List[MarketSegment]:
        return [item for item in self.user_segment_dist.draw_n(num_items, replace=True)]

    def print_day_summary(self, day: int) -> None:
        print(f"\n\t################# DAY {day} SUMMARY ##################")
        print("\n\t#### Agent \t\t# Profit \t###")
        print("\n\t###########################################")
        for agent in self.agents:
            agent_state = self.states[agent]
            print(f"\n\t### {agent.name:12} \t# {agent_state.profits:8.2f} \t###")
        print("\n\t###########################################")

    def print_game_results(self) -> None:
        print("\n\t################# SIMULATION RESULTS ##################")
        print("\n\t#### Agent \t\t# Profit \t###")
        print("\n\t###########################################")
        for agent in sorted(self.agents, key=lambda a: self.states[a].profits, reverse=True):
            agent_state = self.states[agent]
            print(f"\n\t### {agent.name:12} \t# {agent_state.profits:8.2f} \t###")
        print("\n\t###########################################")
        winner = max(self.agents, key=lambda a: self.states[a].profits)
        print(f"\n\t@@@ WINNER: {winner.name} @@@")
    
    def print_final_results(self, total_profits : Dict[NDaysNCampaignsAgent, float], num_simulations: int) -> None:
        print("\n\t################# SIMULATION RESULTS ##################")
        print("\n\t#### Agent \t\t# Profit \t###")
        print("\n\t###########################################")
        for agent in sorted(self.agents, key=lambda a: total_profits[a.name], reverse=True):
            print(f"\n\t### {agent.name:12} \t# {total_profits[agent.name]/num_simulations:8.2f} \t###")
        print("\n\t###########################################")
        winner = max(self.agents, key=lambda a: total_profits[a.name])
        print(f"\n\t@@@ WINNER: {winner.name} @@@")

    def run_simulation(self, agents: list[NDaysNCampaignsAgent], num_simulations: int) -> None:
        total_profits = {agent.name : 0.0 for agent in agents}
        for i in range(num_simulations):
            self.states = self.init_agents(agents)
            self.campaigns = dict()
            # Initialize campaigns 
            for agent in self.agents:    
                    agent.current_game = i + 1 
                    agent.my_campaigns = set()
                    agent.on_new_game()
                    random_campaign = self.generate_campaign(start_day=1)
                    agent_state = self.states[agent]
                    random_campaign.budget = random_campaign.reach
                    agent_state.add_campaign(random_campaign)
                    agent.my_campaigns.add(random_campaign)
                    self.campaigns[random_campaign.uid] = random_campaign

            for day in range(1, self.num_days + 1):
                # Update 
                for agent in self.agents:
                    agent.current_day = day

                # Generate new campaigns and filter
                if day + 1 < self.num_days + 1:
                    new_campaigns = [self.generate_campaign(start_day=day + 1) for _ in range(self.campaigns_per_day)]
                    new_campaigns = [c for c in new_campaigns if c.end_day <= self.num_days]
                    # Solicit campaign bids and run campaign auctions
                    agent_bids = dict()
                    for agent in self.agents:
                        agent_bids[agent] = agent.get_campaign_bids(new_campaigns)

                # Solicit ad bids from agents and run ad auctions
                ad_bids = []
                for agent in self.agents:
                    ad_bids.extend(agent.get_ad_bids())
                users = self.generate_auction_items(10000)
                self.run_ad_auctions(ad_bids, users, day)

                # Update campaign states, quality scores, and profits
                for agent in self.agents:
                    agent_state = self.states[agent]
                    todays_profit = 0.0
                    new_qs_count = 0
                    new_qs_val = 0.0

                    for campaign in agent_state.campaigns.values():
                        if campaign.start_day <= day <= campaign.end_day:
                            if day == campaign.end_day:
                                impressions = agent_state.impressions[campaign.uid]
                                total_cost = agent_state.spend[campaign.uid]
                                effective_reach = calculate_effective_reach(impressions, campaign.reach)
                                todays_profit += (effective_reach) * agent_state.budgets[campaign.uid] - total_cost

                                new_qs_count += 1
                                new_qs_val += effective_reach

                    if new_qs_count > 0:
                        new_qs_val /= new_qs_count
                        self.states[agent].quality_score = (1 - self.α) * self.states[agent].quality_score + self.α * new_qs_val
                        agent.quality_score = self.states[agent].quality_score

                    agent_state.profits += todays_profit
                    agent.profit += todays_profit
                
                # Run campaign auctions
                self.run_campaign_auctions(agent_bids, new_campaigns)
                # Run campaign endowments
                for agent in self.agents:
                    if random.random() < min(1, agent.quality_score):
                        random_campaign = self.generate_campaign(start_day=day)
                        agent_state = self.states[agent]
                        random_campaign.budget = random_campaign.reach
                        agent_state.add_campaign(random_campaign)
                        agent.my_campaigns.add(random_campaign)
                        self.campaigns[random_campaign.uid] = random_campaign
            for agent in self.agents:
                total_profits[agent.name] += self.states[agent].profits 
            self.print_game_results()
        self.print_final_results(total_profits, num_simulations)
        return total_profits
