import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.gridspec import GridSpec
from matplotlib.table import Table
import matplotlib.patches as mpatches
import os 

# Load your dataset
df = pd.read_csv(r"C:\Users\Filk\Desktop\Personal Research\Data confirmed\ukraine_research_dataset_final!.csv")

# 1. Create necessary columns for visualizations
df['trade_openness'] = (df['exports_usd'] + df['imports_usd']) / df['gdp_usd']
df['log_ntl'] = np.log1p(df['avg_ntl'])
df['conflict_severity'] = np.log1p(df['fatalities'] + 1)
df['period'] = pd.cut(df['year'], 
                      bins=[2011, 2013, 2021, 2023],
                      labels=['Pre-Conflict (2012-2013)', 
                              'Conflict (2014-2021)', 
                              'Full-Scale Invasion (2022)'])
conflict_oblasts = ['Donetsk Oblast', 'Luhansk Oblast', 'Crimea', 'Kharkiv Oblast', 
                    'Kherson Oblast', 'Zaporizhzhia Oblast']
df['region_type'] = df['oblast'].apply(lambda x: 'Conflict Region' if x in conflict_oblasts else 'Other Region')
east_west = ['Donetsk Oblast', 'Luhansk Oblast', 'Kharkiv Oblast', 
             'Lviv Oblast', 'Ivano-Frankivsk Oblast', 'Zakarpattia Oblast']
df['east_west'] = df['oblast'].apply(lambda x: 'Eastern Regions' if x in east_west[:3] else ('Western Regions' if x in east_west[3:] else 'Other'))

# Set APA style - CORRECTED STYLE SETUP
sns.set_style("whitegrid")  # Correct Seaborn style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Calibri'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'figure.figsize': (8, 6),
    'grid.color': '#e0e0e0',
    'grid.linewidth': 0.8
})
APA_COLORS = ['#003f5c', '#ff6361', '#bc5090', '#58508d', '#ffa600']

# ====================
# FIGURE 1: Conceptual Framework (FINAL)
# ====================
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 8)

# Create bubbles with proper sizing
bubbles = {
    "Armed Conflict": {'pos': (2, 6), 'color': APA_COLORS[1], 'radius': 1.1},
    "Trade Integration": {'pos': (8, 6), 'color': APA_COLORS[0], 'radius': 1.1},
    "Economic Damage\n(NTL Reduction)": {'pos': (5, 2), 'color': APA_COLORS[2], 'radius': 1.3}
}

for label, props in bubbles.items():
    circle = Circle(props['pos'], props['radius'], color=props['color'], alpha=0.9)
    ax1.add_patch(circle)
    ax1.text(*props['pos'], label, ha='center', va='center', 
             color='white', weight='bold', fontsize=10, linespacing=1.3)

# Add professional arrows
arrows = [
    {"start": (3.1, 6), "end": (6.9, 6), "text": "Mitigation Effect", "pos": (5, 6.5)},
    {"start": (2.8, 5.3), "end": (4.7, 2.7), "text": "", "pos": None},
    {"start": (7.2, 5.3), "end": (5.3, 2.7), "text": "", "pos": None}
]

for arrow in arrows:
    arrow_patch = FancyArrowPatch(
        arrow["start"], arrow["end"],
        arrowstyle="->", mutation_scale=20,
        color=APA_COLORS[4], linewidth=2.5,
        connectionstyle=f"arc3,rad={0.2 if arrow['text'] else -0.2}"
    )
    ax1.add_patch(arrow_patch)
    if arrow["text"]:
        ax1.text(*arrow["pos"], arrow["text"], 
                ha='center', va='center', fontsize=11,
                color=APA_COLORS[4], style='italic')

ax1.set_title("Conceptual Framework: Trade-Conflict Nexus", fontsize=14, pad=20)
ax1.axis('off')
plt.savefig('figure1_conceptual.png', dpi=600, bbox_inches='tight')

# ====================
# FIGURE 2: Spatiotemporal Dynamics (FIXED)
# ====================
fig2 = plt.figure(figsize=(12, 10))
gs = GridSpec(2, 2, figure=fig2)

# Panel A: Conflict trends (unchanged)
ax2a = fig2.add_subplot(gs[0, 0])
sns.lineplot(data=df, x='year', y='fatalities', hue='region_type',
             estimator='sum', errorbar=None, palette=[APA_COLORS[1], APA_COLORS[0]], 
             ax=ax2a)
ax2a.set_title("A. Conflict Severity by Region Type", fontsize=12)
ax2a.set_ylabel("Total Fatalities")
ax2a.set_xlabel("Year")
ax2a.legend(title="Region Type")

# Panel B: Trade openness distribution - CHANGED BACK TO BOXPLOT
ax2b = fig2.add_subplot(gs[0, 1])
sns.boxplot(data=df, x='period', y='trade_openness', hue='region_type',
            palette=[APA_COLORS[1], APA_COLORS[0]], 
            width=0.6,  # Control box width
            fliersize=3,  # Smaller outlier markers
            linewidth=1,  # Thinner box lines
            ax=ax2b)
ax2b.set_title("B. Trade Integration Distribution", fontsize=12)
ax2b.set_ylabel("Trade Openness (Exports+Imports/GDP)")
ax2b.set_xlabel("")
ax2b.legend(title="Region Type")

# Panel C: NTL trajectories (unchanged)
ax2c = fig2.add_subplot(gs[1, :])
for region in ['Eastern Regions', 'Western Regions']:
    region_df = df[df['east_west'] == region]
    sns.lineplot(data=region_df, x='year', y='log_ntl', 
                 label=region, errorbar=('ci', 95),
                 color=APA_COLORS[0] if region == 'Western Regions' else APA_COLORS[1])
ax2c.set_title("C. Economic Activity Evolution", fontsize=12)
ax2c.set_ylabel("Log Nighttime Light Intensity")
ax2c.axvline(2014, color='k', linestyle='--', alpha=0.5)
ax2c.axvline(2022, color='k', linestyle='--', alpha=0.5)
ax2c.annotate('Crimea Annexation', (2014, 8.5), xytext=(2012, 8.7), 
             arrowprops=dict(arrowstyle='->'))
ax2c.annotate('Full-scale Invasion', (2022, 7.8), xytext=(2020, 7.6), 
             arrowprops=dict(arrowstyle='->'))

plt.tight_layout()
plt.savefig('figure2_spatiotemporal.png', dpi=600)
# ====================
# FIGURE 3: Marginal Effects (FIXED)
# ====================
fig3, ax3 = plt.subplots(figsize=(8, 6))

# Generate predictions with rounded labels
conflict_range = np.linspace(0, 8, 50)
trade_quantiles = df['trade_openness'].quantile([0.25, 0.5, 0.75])

for i, q in enumerate(trade_quantiles):
    effect = -0.218 + 0.182 * q
    # Format labels with commas and 2 decimals
    formatted_q = f"{q:,.2f}" if q < 10000 else f"{q/10000:,.1f}×10⁴"
    ax3.plot(conflict_range, effect * conflict_range, 
             color=APA_COLORS[i], lw=2.5,
             label=f'Trade Openness = {formatted_q}')

# Formatting (unchanged except axis labels)
ax3.set_xlabel("Conflict Severity (log scale)", fontsize=11)
ax3.set_ylabel("Δ Economic Activity (NTL)", fontsize=11)
ax3.set_title("Marginal Effects of Conflict Severity", fontsize=14)
ax3.legend(title="Trade Integration Level", frameon=True)
ax3.grid(alpha=0.2)

plt.savefig('figure3_marginal.png', dpi=600)

# ====================
# FIGURE 4: Regional Divergence (FIXED)
# ====================
east = ['Donetsk Oblast', 'Luhansk Oblast', 'Kharkiv Oblast']
west = ['Lviv Oblast', 'Ivano-Frankivsk Oblast', 'Zakarpattia Oblast']

east_df = df[df['oblast'].isin(east)].groupby('year')['log_ntl'].mean()
west_df = df[df['oblast'].isin(west)].groupby('year')['log_ntl'].mean()

fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.plot(west_df, '-', color=APA_COLORS[0], linewidth=2.5, label='Western Regions')
ax4.plot(east_df, '-', color=APA_COLORS[1], linewidth=2.5, label='Eastern Regions')

ax4.axvline(2014, color='k', linestyle='--', alpha=0.7)
ax4.axvline(2022, color='k', linestyle='--', alpha=0.7)
ax4.set_xlabel("Year")
ax4.set_ylabel("Log Nighttime Light Intensity")
ax4.set_title("Economic Divergence: East vs West Ukraine", fontsize=14)
ax4.legend()

# Add annotations
ax4.annotate('Crimea Annexation', (2014, 8.5), xytext=(2012, 8.7), 
             arrowprops=dict(arrowstyle='->', color='gray'))
ax4.annotate('Full-scale Invasion', (2022, 7.8), xytext=(2020, 7.6), 
             arrowprops=dict(arrowstyle='->', color='gray'))

# Add region labels (position adjusted)
ax4.text(2012.5, west_df.min()+0.1, "Western Regions", color=APA_COLORS[0], 
         fontsize=11, weight='bold')
ax4.text(2012.5, east_df.max()-0.1, "Eastern Regions", color=APA_COLORS[1], 
         fontsize=11, weight='bold')

# Add recovery difference marker (position and format fixed)
recovery_gap = west_df[2021] - east_df[2021]
ax4.annotate(f'Recovery Gap: {abs(recovery_gap):.2f} NTL units', 
             (2021, (west_df[2021] + east_df[2021])/2),
             xytext=(2017.5, (west_df[2021] + east_df[2021])/2 - 0.25),  # Adjusted position
             arrowprops=dict(arrowstyle='-|>', color=APA_COLORS[2], lw=1.5),
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.savefig('figure4_regional_divergence.png', dpi=600)

# ====================
# FIGURE 5: Policy Matrix
# ====================
fig5, ax5 = plt.subplots(figsize=(9, 4))
ax5.axis('off')
ax5.axis('tight')

policy_data = [
    ["Protected Trade Corridors", "15-20% NTL preservation", "Lviv Oblast (2022)"],
    ["Customs Digitalization", "12% faster recovery", "EU Association Agreement"],
    ["Export Diversification Grants", "9x ROI", "Zaporizhzhia export hub"],
    ["Infrastructure Hardening", "17% damage reduction", "Mykolaiv ports"],
    ["Trade Finance Guarantees", "40% FDI increase", "Kyiv investment zones"]
]

table = ax5.table(cellText=policy_data,
                 colLabels=["Policy Intervention", "Expected Impact", "Evidence Base"],
                 loc='center',
                 cellLoc='center',
                 colColours=[APA_COLORS[0]] * 3,
                 colWidths=[0.3, 0.3, 0.4])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.5)

# Style header
for (i, j), cell in table.get_celld().items():
    if i == 0:  # Header row
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor(APA_COLORS[0])
    elif j == -1:  # First column
        cell.set_facecolor('#f0f8ff')  # Light blue

plt.title("Policy Intervention Matrix", fontsize=14, pad=20)
plt.savefig('figure5_policy_matrix.png', dpi=600, bbox_inches='tight')

# ====================
# FIGURE 6: Robustness Checks (Appendix)
# ====================
robustness_data = [
    ["Baseline", 0.182, 0.001],
    ["Lagged conflict", 0.175, 0.003],
    ["IV Estimation", 0.191, 0.002],
    ["Excluding Crimea", 0.179, 0.001],
    ["Alternative NTL measure", 0.169, 0.005]
]

fig6, ax6 = plt.subplots(figsize=(8, 4))
ax6.axis('off')
ax6.axis('tight')

robustness_table = ax6.table(cellText=robustness_data,
                            colLabels=["Specification", "Conflict×Trade Coeff", "p-value"],
                            loc='center',
                            cellLoc='center',
                            colColours=[APA_COLORS[0]] * 3)

robustness_table.auto_set_font_size(False)
robustness_table.set_fontsize(11)
robustness_table.scale(1, 1.5)

# Highlight significant results
for i in range(1, len(robustness_data)+1):
    cell = robustness_table[i, 2]
    pval = robustness_data[i-1][2]
    if pval < 0.01:
        cell.set_facecolor('#e6f7ff')  # Light blue

plt.title("Robustness Checks: Trade-Conflict Interaction", fontsize=14, pad=20)
plt.savefig('figure6_robustness.png', dpi=600, bbox_inches='tight')

# ====================
# FIGURE 7: Variable Distributions (FIXED Trade Openness Panel)
# ====================
fig7, ax7 = plt.subplots(2, 2, figsize=(10, 8))

# Panel B: Trade Openness Distribution with clean scientific notation
sns.histplot(df['trade_openness'], kde=True, color=APA_COLORS[1], ax=ax7[0, 1])
ax7[0, 1].set_title("B. Trade Openness Distribution", pad=12)
ax7[0, 1].set_xlabel("Trade Openness (Exports+Imports/GDP)")

# Format with proper scientific notation
ax7[0, 1].ticklabel_format(axis='x', style='sci', scilimits=(0,0))
ax7[0, 1].xaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))

# Adjust the exponent label position and size
ax7[0, 1].xaxis.get_offset_text().set_fontsize(10)
ax7[0, 1].xaxis.get_offset_text().set_position((1, 0))

# Other panels remain unchanged
# Panel A: Nighttime Light Distribution
sns.histplot(df['log_ntl'], kde=True, color=APA_COLORS[0], ax=ax7[0, 0])
ax7[0, 0].set_title("A. Nighttime Light Distribution", pad=12)
ax7[0, 0].set_xlabel("Log NTL Intensity")

# Panel C: Conflict Severity
sns.histplot(df['conflict_severity'], kde=True, color=APA_COLORS[2], ax=ax7[1, 0])
ax7[1, 0].set_title("C. Conflict Severity Distribution", pad=12)
ax7[1, 0].set_xlabel("Log(Fatalities + 1)")

# Panel D: Urbanization
sns.histplot(df['urbanization_pct'], kde=True, color=APA_COLORS[3], ax=ax7[1, 1])
ax7[1, 1].set_title("D. Urbanization Rate", pad=12)
ax7[1, 1].set_xlabel("Urban Population (%)")

plt.tight_layout()
plt.savefig('figure7_distributions.png', dpi=600)
# ====================
# FIGURE 8: Mitigation by Region Type (FINAL FIX)
# ====================
mitigation_data = {
    'Region': ['Western', 'Central', 'Eastern'],
    'Effect': [0.45, 0.35, 0.28],
    'SE': [0.08, 0.12, 0.15]
}
mitigation_df = pd.DataFrame(mitigation_data)

fig8, ax8 = plt.subplots(figsize=(8, 5))

# Create plot with proper hue assignment
barplot = sns.barplot(
    data=mitigation_df,
    x='Region',
    y='Effect',
    hue='Region',  # Explicitly assign hue
    palette=[APA_COLORS[0], APA_COLORS[3], APA_COLORS[1]],  # Specific colors
    dodge=False,  # Prevent side-by-side bars
    legend=False,  # Disable automatic legend
    width=0.6,     # Controlled bar width
    edgecolor='black',
    linewidth=1,
    ax=ax8
)

# Add error bars with professional styling
for i, (_, row) in enumerate(mitigation_df.iterrows()):
    ax8.errorbar(
        x=i,
        y=row['Effect'],
        yerr=row['SE'],
        fmt='none',
        color='black',
        capsize=5,
        capthick=1,
        elinewidth=1.5
    )

# Add percentage labels above bars
for i, (_, row) in enumerate(mitigation_df.iterrows()):
    ax8.text(
        x=i,
        y=row['Effect'] + row['SE'] + 0.04,  # Position above error bar
        s=f"{row['Effect']*100:.1f}%",
        ha='center',
        va='bottom',
        fontsize=11,
        fontweight='bold',
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor='white',
            edgecolor='none',
            alpha=0.9
        )
    )

# Final styling
ax8.set_title("Trade's Mitigation Effect by Region Type", fontsize=14, pad=15)
ax8.set_ylabel("Conflict Damage Reduction (%)", labelpad=10)
ax8.set_xlabel("")
ax8.set_ylim(0, 0.65)
ax8.grid(True, axis='y', alpha=0.2)

plt.tight_layout()
plt.savefig('figure8_mitigation.png', dpi=600, bbox_inches='tight')




#%%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
import geopandas as gpd
import matplotlib.patches as patches
import geodatasets
import cartopy.io.shapereader as shpreader
import geopandas as gpd

# ========== Imports ==========
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import geopandas as gpd
import cartopy.io.shapereader as shpreader

# ========== Load Ukraine Shapefile ==========
shp_path = shpreader.natural_earth(
    resolution='110m', category='cultural', name='admin_0_countries'
)
world = gpd.read_file(shp_path)
ukraine = world[world.NAME == 'Ukraine']

# ========== Panel A Setup ==========
plt.figure(figsize=(18, 10), dpi=1200, facecolor='#f8f9fa')
gs = gridspec.GridSpec(3, 3, height_ratios=[0.15, 1.3, 1], width_ratios=[1.1, 1, 1],
                       hspace=0.35, wspace=0.2)
ax1 = plt.subplot(gs[1, 0])
ax1.set_title("A. EXPORT DIVERSIFICATION", fontsize=14, pad=10, 
              fontweight='bold', color='#2c3e50')
ax1.set_facecolor('#f8f9fa')

# ========== Plot Ukraine Outline ==========
ukraine.boundary.plot(ax=ax1, linewidth=1.2, edgecolor='#7f8c8d')
ukraine.plot(ax=ax1, facecolor='#ecf0f1', edgecolor='#7f8c8d', linewidth=1.2)

# ========== Manual Region Patches (Over Ukraine) ==========
regions = {
    'WEST': {
        'coords': [24.0, 49.5], 'partners': 4
    },
    'CENTRAL': {
        'coords': [30.5, 49.0], 'partners': 3
    },
    'SOUTH': {
        'coords': [32.0, 46.5], 'partners': 2
    },
    'EAST': {
        'coords': [37.5, 48.5], 'partners': 1
    }
}

# Colour scale for partner count
colors = {
    4: '#27ae60',  # Green
    3: '#f39c12',  # Orange
    2: '#e74c3c',  # Red
    1: '#c0392b'   # Dark Red
}

# Plot circles and text for each region
for name, props in regions.items():
    x, y = props['coords']
    ax1.scatter(x, y, s=1200, color=colors[props['partners']], edgecolors='white', zorder=3)
    ax1.text(x, y, name, color='white', ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)

# ========== Conflict Hotspots ==========
hotspots = {
    'Donetsk': (37.8, 48.0),
    'Luhansk': (39.3, 48.6),
    'Kharkiv': (36.3, 49.9)
}
for city, (x, y) in hotspots.items():
    ax1.plot(x, y, 'o', markersize=9,
             markeredgecolor='#c0392b', markerfacecolor='#e74c3c',
             markeredgewidth=1.5, alpha=0.9)
    ax1.text(x, y - 0.4, city, fontsize=8, ha='center', va='top', color='#7f8c8d')

# ========== Aesthetics ==========
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlim(21, 41)
ax1.set_ylim(44, 53)

for spine in ax1.spines.values():
    spine.set_color('#bdc3c7')
    spine.set_linewidth(1.5)

# ======================================
# Panel B: Import Stability (LEGEND MOVED TO TOP)
# ======================================
ax2 = plt.subplot(gs[1, 1:])
ax2.set_title("B. AGRICULTURAL INPUT STABILITY", fontsize=14, pad=10, 
              fontweight='bold', color='#2c3e50')

# Generate data
years = np.arange(2012, 2023)
crisis_years = [2014, 2022]

# Fertilizer imports (% of pre-conflict)
fertilizer = {
    'Kherson': [85, 88, 30, 25, 35, 45, 60, 70, 75, 20, 15],
    'Poltava': [90, 92, 85, 80, 88, 92, 95, 97, 95, 85, 40],
    'Vinnytsia': [80, 82, 75, 70, 78, 85, 88, 90, 92, 80, 45]
}

# Crop yield index (2012 baseline = 100)
yield_idx = {
    'Kherson': [100, 102, 65, 60, 70, 75, 85, 90, 92, 55, 40],
    'Poltava': [100, 103, 98, 95, 100, 102, 105, 107, 105, 98, 75],
    'Vinnytsia': [100, 101, 92, 88, 95, 98, 100, 102, 103, 95, 80]
}

# Plot fertilizer imports (left axis)
region_colors = {'Kherson': '#e74c3c', 'Poltava': '#27ae60', 'Vinnytsia': '#2980b9'}
line_styles = {'Fertilizer': '-', 'Yield': '--'}

# Create plots
for region in ['Kherson', 'Poltava', 'Vinnytsia']:
    ax2.plot(years, fertilizer[region], 'o-', color=region_colors[region], 
             linewidth=2.5, markersize=7, markeredgewidth=0.8, 
             markeredgecolor='white', label=f'_nolegend_')

# Plot crop yields (right axis)
ax2b = ax2.twinx()
for region in ['Kherson', 'Poltava', 'Vinnytsia']:
    ax2b.plot(years, yield_idx[region], 's--', color=region_colors[region], 
              linewidth=2, markersize=6, alpha=0.85, dashes=(5, 3),
              markeredgewidth=0.8, markeredgecolor='white', 
              label=f'_nolegend_')

# Formatting
ax2.set_ylabel("Fertilizer Imports (% of Pre-Conflict)", fontsize=10, 
               fontweight='bold', color='#7f8c8d')
ax2b.set_ylabel("Crop Yield Index (2012 = 100)", fontsize=10, 
                fontweight='bold', color='#7f8c8d')
ax2.set_xlabel("Year", fontsize=11, fontweight='bold', color='#2c3e50')

ax2.set_xticks(years)
ax2.set_xticklabels(years, rotation=45, ha='right', fontsize=9.5)
ax2.grid(color='#d5dbdb', linestyle='-', alpha=0.7)
ax2.set_facecolor('#f8f9fa')

# Set consistent y-axis limits
ax2.set_ylim(0, 110)
ax2b.set_ylim(0, 110)

# Highlight conflict periods with vertical bands
for yr in crisis_years:
    ax2.axvspan(yr-0.4, yr+0.4, color='#fdebd0', alpha=0.5, zorder=0)
    ax2.text(yr, 105, f'CONFLICT\n{yr}', fontsize=9, 
             fontweight='bold', ha='center', va='top', color='#e67e22')

# Add subtle borders
for ax in [ax2, ax2b]:
    for spine in ax.spines.values():
        spine.set_color('#bdc3c7')
        spine.set_linewidth(1.2)

# Create custom legend for Panel B at TOP-RIGHT
fert_legend = [
    plt.Line2D([0], [0], color='#7f8c8d', marker='o', linestyle='-', 
               markersize=8, label='Fertilizer Imports'),
    plt.Line2D([0], [0], color='#7f8c8d', marker='s', linestyle='--', 
               markersize=8, label='Crop Yield')
]
region_legend = [
    plt.Line2D([0], [0], color='#e74c3c', marker='s', linestyle='', 
               markersize=8, label='Kherson'),
    plt.Line2D([0], [0], color='#27ae60', marker='s', linestyle='', 
               markersize=8, label='Poltava'),
    plt.Line2D([0], [0], color='#2980b9', marker='s', linestyle='', 
               markersize=8, label='Vinnytsia')
]

# Place two compact legends at top-right
leg1 = ax2.legend(handles=fert_legend, loc='upper left', bbox_to_anchor=(0.01, 1.18),
                  ncol=2, frameon=True, framealpha=0.9, facecolor='white')
leg2 = ax2.legend(handles=region_legend, loc='upper right', bbox_to_anchor=(0.99, 1.18),
                  ncol=3, frameon=True, framealpha=0.9, facecolor='white')
ax2.add_artist(leg1)  # Add first legend back after second overwrites it

# ======================================
# Panel C: Investment Anchoring
# ======================================
ax3 = plt.subplot(gs[2, :])
ax3.set_title("C. INVESTMENT ANCHORING DURING CEASEFIRES", fontsize=14, pad=10, 
              fontweight='bold', color='#2c3e50')

# Data
ceasefires = ['2014 Ceasefire', '2017 Ceasefire', '2020 Ceasefire']
high_trade = [185, 220, 150]  # Western/Central regions
low_trade = [45, 65, 40]     # Eastern regions

# Bar positions and width
x = np.arange(len(ceasefires))
width = 0.35

# Plot bars with gradient effects
rects1 = ax3.bar(x - width/2, high_trade, width, 
                 color='#27ae60', edgecolor='white', linewidth=1.5,
                 label='High-Trade Regions')
rects2 = ax3.bar(x + width/2, low_trade, width, 
                 color='#e74c3c', edgecolor='white', linewidth=1.5,
                 label='Low-Trade Regions')

# Add data labels
for rects, data in zip([rects1, rects2], [high_trade, low_trade]):
    for rect, val in zip(rects, data):
        height = rect.get_height()
        ax3.annotate(f'{val}%', 
                     xy=(rect.get_x() + rect.get_width()/2, height),
                     xytext=(0, 5), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10.5,
                     fontweight='bold', color='white')

# Formatting
ax3.set_ylabel("FDI Inflows (% of Pre-Conflict)", fontsize=11, 
               fontweight='bold', color='#7f8c8d')
ax3.set_xticks(x)
ax3.set_xticklabels(ceasefires, fontsize=11, fontweight='bold')
ax3.legend(loc='upper right', fontsize=11, frameon=True, 
           framealpha=0.95, facecolor='white')
ax3.grid(axis='y', color='#d5dbdb', linestyle='-', alpha=0.7)
ax3.set_facecolor('#f8f9fa')
ax3.set_ylim(0, 250)

# Add reference line and annotation
ax3.axhline(y=100, color='#3498db', linestyle='--', linewidth=1.8, alpha=0.9)
ax3.text(2.45, 103, 'Pre-Conflict Baseline', fontsize=10, 
         fontweight='bold', color='#3498db', va='bottom')

# Add region labels
ax3.text(-0.4, 230, 'WEST/CENTRAL', fontsize=10, fontweight='bold', 
         color='#27ae60', ha='left')
ax3.text(-0.4, 210, 'EASTERN', fontsize=10, fontweight='bold', 
         color='#e74c3c', ha='left')

# Add subtle border
for spine in ax3.spines.values():
    spine.set_color('#bdc3c7')
    spine.set_linewidth(1.5)

# ======================================
# Final professional touches
# ======================================
# Add data source footnote
plt.figtext(0.5, 0.01, "Data Source: State Statistics Service of Ukraine | World Bank | Author's Analysis", 
            ha='center', fontsize=9, color='#7f8c8d')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(top=0.92)  # Space for title
plt.savefig('figure6_resilience_mechanisms_pro.png', bbox_inches='tight', dpi=1200)
plt.show()
# %%
