"""
College Basketball Team Branding
Dynamic logo/color fetching via ESPN API data (stored per-game)
Unlike NBA with fixed 30 teams, college has 360+ teams so we use dynamic lookups.
"""

# Cache for team data fetched at runtime
_team_cache = {}


def get_team_logo(team_id, game_data=None):
    """Get team logo URL - pulled from ESPN game data"""
    if team_id in _team_cache and 'logo' in _team_cache[team_id]:
        return _team_cache[team_id]['logo']
    
    if game_data:
        # Extract from game data
        for prefix in ['home', 'visitor']:
            if game_data.get(f'{prefix}_team_id') == team_id:
                logo = game_data.get(f'{prefix}_team_logo', '')
                if logo:
                    _update_cache(team_id, logo=logo)
                    return logo
    
    # Fallback ESPN logo URL pattern
    return f"https://a.espncdn.com/i/teamlogos/ncaa/500/{team_id}.png"


def get_team_name(team_id, game_data=None):
    """Get team display name"""
    if team_id in _team_cache and 'name' in _team_cache[team_id]:
        return _team_cache[team_id]['name']
    
    if game_data:
        for prefix in ['home', 'visitor']:
            if game_data.get(f'{prefix}_team_id') == team_id:
                name = game_data.get(f'{prefix}_team_name', '')
                if name:
                    _update_cache(team_id, name=name)
                    return name
    
    return f"Team {team_id}"


def get_team_abbrev(team_id, game_data=None):
    """Get team abbreviation"""
    if team_id in _team_cache and 'abbrev' in _team_cache[team_id]:
        return _team_cache[team_id]['abbrev']
    
    if game_data:
        for prefix in ['home', 'visitor']:
            if game_data.get(f'{prefix}_team_id') == team_id:
                abbrev = game_data.get(f'{prefix}_team_abbreviation', '')
                if abbrev:
                    _update_cache(team_id, abbrev=abbrev)
                    return abbrev
    
    return "UNK"


def get_team_color(team_id, game_data=None):
    """Get team primary color"""
    if team_id in _team_cache and 'color' in _team_cache[team_id]:
        return _team_cache[team_id]['color']
    
    if game_data:
        for prefix in ['home', 'visitor']:
            if game_data.get(f'{prefix}_team_id') == team_id:
                color = game_data.get(f'{prefix}_team_color', '444444')
                if color:
                    _update_cache(team_id, color=f"#{color}" if not color.startswith('#') else color)
                    return _team_cache[team_id]['color']
    
    return "#444444"


def _update_cache(team_id, **kwargs):
    """Update team cache with new data"""
    if team_id not in _team_cache:
        _team_cache[team_id] = {}
    _team_cache[team_id].update(kwargs)


def cache_team_from_game(game_data):
    """Pre-cache team data from a game dict"""
    for prefix in ['home', 'visitor']:
        team_id = game_data.get(f'{prefix}_team_id')
        if team_id:
            _update_cache(
                team_id,
                name=game_data.get(f'{prefix}_team_name', ''),
                abbrev=game_data.get(f'{prefix}_team_abbreviation', ''),
                logo=game_data.get(f'{prefix}_team_logo', ''),
                color=game_data.get(f'{prefix}_team_color', '444444'),
                record=game_data.get(f'{prefix}_team_record', ''),
            )
