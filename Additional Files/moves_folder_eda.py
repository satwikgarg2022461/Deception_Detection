#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis on the Diplomacy Game Moves Folder
This script analyzes the game moves data from the 12 Diplomacy games
that were not used in the original paper.
"""

import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from tqdm import tqdm

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Define paths
project_dir = r""
moves_dir = os.path.join(project_dir, "moves")

def extract_game_info(filename):
    """Extract game number, year, and season from filename."""
    pattern = r"DiplomacyGame(\d+)_(\d+)_(spring|fall|winter)\.json"
    match = re.match(pattern, filename)
    if match:
        game_num = int(match.group(1))
        year = int(match.group(2))
        season = match.group(3)
        return game_num, year, season
    return None, None, None

def load_moves_data():
    """Load all move data from the moves directory."""
    moves_data = []

    for filename in tqdm(os.listdir(moves_dir), desc="Loading move files"):
        game_num, year, season = extract_game_info(filename)
        if game_num is None:
            continue

        file_path = os.path.join(moves_dir, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract supply center counts
            sc_text = data.get('sc', '')
            sc_counts = {}

            for line in sc_text.split('\n'):
                line = line.strip()
                match = re.search(r'(\w+)\s+(\d+)', line)
                if match:
                    country = match.group(1)
                    count = int(match.group(2))
                    sc_counts[country] = count

            moves_entry = {
                'game_num': game_num,
                'year': year,
                'season': season,
                'sc_counts': sc_counts,
                'orders': data.get('orders', {}),
                'territories': data.get('territories', {})
            }

            moves_data.append(moves_entry)

        except Exception as e:
            # print(f"Error loading {filename}: {e}")
            print("Error loading file:", filename)

    return moves_data

def analyze_order_types(moves_data):
    order_types = []

    for move in moves_data:
        for country, orders in move['orders'].items():
            for location, order in orders.items():
                order_type = order.get('type', 'UNKNOWN')
                order_types.append(order_type)

    order_counts = Counter(order_types)

    df_orders = pd.DataFrame({
        'Order Type': list(order_counts.keys()),
        'Count': list(order_counts.values())
    }).sort_values('Count', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Order Type', y='Count', data=df_orders)
    plt.title('Distribution of Order Types')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(project_dir, 'order_types_distribution.png'))
    plt.close()

    return df_orders

def analyze_order_results(moves_data):
    order_results = []

    for move in moves_data:
        for country, orders in move['orders'].items():
            for location, order in orders.items():
                order_result = order.get('result', 'UNKNOWN')
                order_results.append(order_result)

    result_counts = Counter(order_results)

    df_results = pd.DataFrame({
        'Result': list(result_counts.keys()),
        'Count': list(result_counts.values())
    }).sort_values('Count', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Result', y='Count', data=df_results)
    plt.title('Distribution of Order Results')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(project_dir, 'order_results_distribution.png'))
    plt.close()

    return df_results

def track_supply_centers_over_time(moves_data):
    sc_by_game = defaultdict(lambda: defaultdict(dict))

    for move in moves_data:
        game = move['game_num']
        year = move['year']
        season = move['season']
        season_value = {'winter': 3, 'fall': 2, 'spring': 1}
        time_key = (year, season_value[season])

        for country, count in move['sc_counts'].items():
            sc_by_game[game][country][time_key] = count

    for game in sorted(sc_by_game.keys()):
        plt.figure(figsize=(12, 8))

        for country in sc_by_game[game]:
            time_keys = sorted(sc_by_game[game][country].keys())
            counts = [sc_by_game[game][country][key] for key in time_keys]

            season_labels = {1: 'Spring', 2: 'Fall', 3: 'Winter'}
            formatted_times = [f"{year} {season_labels[season]}" for year, season in time_keys]

            plt.plot(formatted_times, counts, marker='o', label=country)

        plt.title(f'Supply Center Counts Over Time - Game {game}')
        plt.xlabel('Time (Year-Season)')
        plt.ylabel('Supply Center Count')
        plt.xticks(rotation=90)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(project_dir, f'supply_centers_game_{game}.png'))
        plt.close()

def analyze_territory_control(moves_data):
    all_territories = set()
    for move in moves_data:
        all_territories.update(move['territories'].keys())

    territory_control = {territory: defaultdict(int) for territory in all_territories}

    for move in moves_data:
        for territory, country in move['territories'].items():
            territory_control[territory][country] += 1

    contested_territories = []

    for territory, controls in territory_control.items():
        countries = len(controls)
        if countries > 1:
            contested_territories.append({
                'Territory': territory,
                'Countries': countries,
                'Total_Appearances': sum(controls.values()),
                'Most_Common': max(controls.items(), key=lambda x: x[1])[0]
            })

    df_contested = pd.DataFrame(contested_territories)
    if not df_contested.empty:
        df_contested = df_contested.sort_values('Countries', ascending=False).head(20)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Territory', y='Countries', data=df_contested)
        plt.title('Top 20 Most Contested Territories')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(project_dir, 'contested_territories.png'))
        plt.close()

    return df_contested

def analyze_country_performance(moves_data):
    final_scores = defaultdict(list)

    for game_num in range(1, 13):
        game_moves = [m for m in moves_data if m['game_num'] == game_num]

        if not game_moves:
            continue

        def sort_key(move):
            season_value = {'winter': 3, 'fall': 2, 'spring': 1}
            return (move['year'], season_value[move['season']])

        game_moves.sort(key=sort_key)

        last_move = game_moves[-1]
        for country, count in last_move['sc_counts'].items():
            final_scores[country].append(count)

    country_stats = []
    for country, scores in final_scores.items():
        country_stats.append({
            'Country': country,
            'Average Score': np.mean(scores),
            'Max Score': np.max(scores),
            'Min Score': np.min(scores),
            'Std Dev': np.std(scores),
            'Games Played': len(scores)
        })

    df_stats = pd.DataFrame(country_stats).sort_values('Average Score', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Country', y='Average Score', data=df_stats)
    plt.title('Average Final Supply Center Count by Country')
    plt.ylabel('Average Supply Centers')
    plt.tight_layout()
    plt.savefig(os.path.join(project_dir, 'country_performance.png'))
    plt.close()

    return df_stats

def main():
    print("Starting Diplomacy Moves Analysis...")
    moves_data = load_moves_data()
    unique_games = set(m['game_num'] for m in moves_data)
    print(f"Loaded {len(moves_data)} move entries from {len(unique_games)} unique games.")

    df_orders = analyze_order_types(moves_data)
    print("Order Type Analysis Completed.\n", df_orders)

    df_results = analyze_order_results(moves_data)
    print("Order Result Analysis Completed.\n", df_results)

    print("Tracking Supply Centers Over Time...")
    track_supply_centers_over_time(moves_data)

    df_contested = analyze_territory_control(moves_data)
    print("Territory Control Analysis Completed.\n", df_contested)

    df_stats = analyze_country_performance(moves_data)
    print("Country Performance Analysis Completed.\n", df_stats)

if __name__ == "__main__":
    main()
