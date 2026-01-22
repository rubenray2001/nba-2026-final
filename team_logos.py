"""
NBA Team Logos and Branding
"""

# NBA Team Logo URLs (official logos)
TEAM_LOGOS = {
    1: "https://cdn.nba.com/logos/nba/1610612737/primary/L/logo.svg",  # Atlanta Hawks
    2: "https://cdn.nba.com/logos/nba/1610612738/primary/L/logo.svg",  # Boston Celtics
    3: "https://cdn.nba.com/logos/nba/1610612751/primary/L/logo.svg",  # Brooklyn Nets
    4: "https://cdn.nba.com/logos/nba/1610612766/primary/L/logo.svg",  # Charlotte Hornets
    5: "https://cdn.nba.com/logos/nba/1610612741/primary/L/logo.svg",  # Chicago Bulls
    6: "https://cdn.nba.com/logos/nba/1610612739/primary/L/logo.svg",  # Cleveland Cavaliers
    7: "https://cdn.nba.com/logos/nba/1610612742/primary/L/logo.svg",  # Dallas Mavericks
    8: "https://cdn.nba.com/logos/nba/1610612743/primary/L/logo.svg",  # Denver Nuggets
    9: "https://cdn.nba.com/logos/nba/1610612765/primary/L/logo.svg",  # Detroit Pistons
    10: "https://cdn.nba.com/logos/nba/1610612744/primary/L/logo.svg",  # Golden State Warriors
    11: "https://cdn.nba.com/logos/nba/1610612745/primary/L/logo.svg",  # Houston Rockets
    12: "https://cdn.nba.com/logos/nba/1610612754/primary/L/logo.svg",  # Indiana Pacers
    13: "https://cdn.nba.com/logos/nba/1610612746/primary/L/logo.svg",  # LA Clippers
    14: "https://cdn.nba.com/logos/nba/1610612747/primary/L/logo.svg",  # Los Angeles Lakers
    15: "https://cdn.nba.com/logos/nba/1610612763/primary/L/logo.svg",  # Memphis Grizzlies
    16: "https://cdn.nba.com/logos/nba/1610612748/primary/L/logo.svg",  # Miami Heat
    17: "https://cdn.nba.com/logos/nba/1610612749/primary/L/logo.svg",  # Milwaukee Bucks
    18: "https://cdn.nba.com/logos/nba/1610612750/primary/L/logo.svg",  # Minnesota Timberwolves
    19: "https://cdn.nba.com/logos/nba/1610612740/primary/L/logo.svg",  # New Orleans Pelicans
    20: "https://cdn.nba.com/logos/nba/1610612752/primary/L/logo.svg",  # New York Knicks
    21: "https://cdn.nba.com/logos/nba/1610612760/primary/L/logo.svg",  # Oklahoma City Thunder
    22: "https://cdn.nba.com/logos/nba/1610612753/primary/L/logo.svg",  # Orlando Magic
    23: "https://cdn.nba.com/logos/nba/1610612755/primary/L/logo.svg",  # Philadelphia 76ers
    24: "https://cdn.nba.com/logos/nba/1610612756/primary/L/logo.svg",  # Phoenix Suns
    25: "https://cdn.nba.com/logos/nba/1610612757/primary/L/logo.svg",  # Portland Trail Blazers
    26: "https://cdn.nba.com/logos/nba/1610612758/primary/L/logo.svg",  # Sacramento Kings
    27: "https://cdn.nba.com/logos/nba/1610612759/primary/L/logo.svg",  # San Antonio Spurs
    28: "https://cdn.nba.com/logos/nba/1610612761/primary/L/logo.svg",  # Toronto Raptors
    29: "https://cdn.nba.com/logos/nba/1610612762/primary/L/logo.svg",  # Utah Jazz
    30: "https://cdn.nba.com/logos/nba/1610612764/primary/L/logo.svg",  # Washington Wizards
}

# Team abbreviation mapping
TEAM_ABBREVIATIONS = {
    1: "ATL", 2: "BOS", 3: "BKN", 4: "CHA", 5: "CHI",
    6: "CLE", 7: "DAL", 8: "DEN", 9: "DET", 10: "GSW",
    11: "HOU", 12: "IND", 13: "LAC", 14: "LAL", 15: "MEM",
    16: "MIA", 17: "MIL", 18: "MIN", 19: "NOP", 20: "NYK",
    21: "OKC", 22: "ORL", 23: "PHI", 24: "PHX", 25: "POR",
    26: "SAC", 27: "SAS", 28: "TOR", 29: "UTA", 30: "WAS"
}

# Team full names
TEAM_NAMES = {
    1: "Atlanta Hawks", 2: "Boston Celtics", 3: "Brooklyn Nets",
    4: "Charlotte Hornets", 5: "Chicago Bulls", 6: "Cleveland Cavaliers",
    7: "Dallas Mavericks", 8: "Denver Nuggets", 9: "Detroit Pistons",
    10: "Golden State Warriors", 11: "Houston Rockets", 12: "Indiana Pacers",
    13: "LA Clippers", 14: "Los Angeles Lakers", 15: "Memphis Grizzlies",
    16: "Miami Heat", 17: "Milwaukee Bucks", 18: "Minnesota Timberwolves",
    19: "New Orleans Pelicans", 20: "New York Knicks", 21: "Oklahoma City Thunder",
    22: "Orlando Magic", 23: "Philadelphia 76ers", 24: "Phoenix Suns",
    25: "Portland Trail Blazers", 26: "Sacramento Kings", 27: "San Antonio Spurs",
    28: "Toronto Raptors", 29: "Utah Jazz", 30: "Washington Wizards"
}

# Team primary colors (for UI styling)
TEAM_COLORS = {
    1: "#E03A3E", 2: "#007A33", 3: "#000000", 4: "#1D1160", 5: "#CE1141",
    6: "#860038", 7: "#00538C", 8: "#0E2240", 9: "#C8102E", 10: "#1D428A",
    11: "#CE1141", 12: "#002D62", 13: "#C8102E", 14: "#552583", 15: "#5D76A9",
    16: "#98002E", 17: "#00471B", 18: "#0C2340", 19: "#0C2340", 20: "#F58426",
    21: "#007AC1", 22: "#0077C0", 23: "#006BB6", 24: "#1D1160", 25: "#E03A3E",
    26: "#5A2D81", 27: "#C4CED4", 28: "#CE1141", 29: "#002B5C", 30: "#002B5C"
}


def get_team_logo(team_id):
    """Get team logo URL by team ID"""
    return TEAM_LOGOS.get(team_id, "")


def get_team_name(team_id):
    """Get team full name by team ID"""
    return TEAM_NAMES.get(team_id, "Unknown Team")


def get_team_abbrev(team_id):
    """Get team abbreviation by team ID"""
    return TEAM_ABBREVIATIONS.get(team_id, "UNK")


def get_team_color(team_id):
    """Get team primary color by team ID"""
    return TEAM_COLORS.get(team_id, "#000000")
