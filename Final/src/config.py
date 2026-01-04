from pathlib import Path

# ==================== PATHS ====================
# Base paths
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "dataset/raw/AI_Developer_Performance.csv"
PROCESSED_DATA_DIR = BASE_DIR / "dataset/processed/AI_Developer_Performance_processed.csv"

# ==================== FEATURE ENGINEERING ====================
FEATURE_TOTAL_TECH_HOURS = 'Total_Tech_Hours'
FEATURE_AI_INTENSITY = 'AI_Intensity'
FEATURE_EFFICIENCY_LOC = 'Efficiency_LOC_per_Hour'
FEATURE_BUG_RATE = 'Bug_Rate_per_1k_LOC'
FEATURE_AI_USER_TYPE = 'AI_User_Type'
FEATURE_COFFEE_GROUP = 'Coffee_Group'
FEATURE_LOAD_ZONE = 'Load_Zone'

# AI User Type labels
AI_USER_TYPE_LABELS = ['Low AI', 'Moderate AI', 'High AI', 'Extreme AI']

# Coffee Group bins, labels
COFFEE_BINS = [-1, 2, 4, 20]
COFFEE_LABELS = ['Low (0-2 cups)', 'Medium (3-4 cups)', 'High (>4 cups)']

# ==================== MODEL SETTINGS ====================
# Train/Test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Random Forest settings
RF_N_ESTIMATORS = 100

# K-Means settings
KMEANS_N_INIT = 10

# ==================== VISUALIZATION SETTINGS ====================
# Figure sizes
FIG_SIZE_SMALL = (8, 6)
FIG_SIZE_MEDIUM = (10, 8)
FIG_SIZE_LARGE = (14, 12)
FIG_SIZE_WIDE = (16, 6)

# Plot style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
COLOR_PALETTE = 'crest'

# Heatmap settings
HEATMAP_CMAP = 'RdBu_r'
HEATMAP_FMT = '.2f'
