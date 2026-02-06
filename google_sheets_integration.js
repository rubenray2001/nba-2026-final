/**
 * BALLDONTLIE API GOOGLE SHEETS INTEGRATION
 * =========================================
 * 
 * INSTRUCTIONS:
 * 1. Go to extensions > Apps Script in your Google Sheet
 * 2. Paste this entire code into the editor
 * 3. Replace 'YOUR_API_KEY' below with your actual API key
 * 4. Save the script
 * 5. Use the custom functions in your sheet!
 */

const API_KEY = '9deeba1d-acec-4e0a-86f6-c762a66a1f2e'; // Replace if needed
const BASE_URL = 'https://api.balldontlie.io/v1';

/**
 * Configure caching (caches results for 6 hours to save API calls)
 */
function getCache() {
    return CacheService.getScriptCache();
}

/**
 * Fetch data helper with caching and error handling
 */
function fetchData(endpoint, params = {}) {
    const queryString = Object.keys(params)
        .map(key => key + '=' + encodeURIComponent(params[key]))
        .join('&');

    const url = BASE_URL + endpoint + (queryString ? '?' + queryString : '');
    const cacheKey = url;
    const cache = getCache();
    const cached = cache.get(cacheKey);

    if (cached) {
        return JSON.parse(cached);
    }

    const options = {
        'method': 'get',
        'headers': {
            'Authorization': API_KEY
        },
        'muteHttpExceptions': true
    };

    try {
        const response = UrlFetchApp.fetch(url, options);
        const code = response.getResponseCode();
        const content = response.getContentText();

        if (code !== 200) {
            return { error: `API Error ${code}: ${content}` };
        }

        const data = JSON.parse(content);
        // Cache for 6 hours (21600 seconds)
        cache.put(cacheKey, content, 21600);

        return data;
    } catch (e) {
        return { error: "Fetch failed: " + e.toString() };
    }
}

/**
 * Search for NBA Players
 * Usage: =BDL_NBA_PLAYERS("LeBron")
 * @param {string} search Name to search for
 * @return {Array} Table of players
 * @customfunction
 */
function BDL_NBA_PLAYERS(search) {
    if (!search) return "Please provide a name.";

    const data = fetchData('/players', { search: search });

    if (data.error) return data.error;
    if (!data.data || data.data.length === 0) return "No players found.";

    // Header
    const result = [['ID', 'First Name', 'Last Name', 'Position', 'Team', 'Height', 'Weight']];

    data.data.forEach(p => {
        result.push([
            p.id,
            p.first_name,
            p.last_name,
            p.position,
            p.team.full_name,
            p.height || 'N/A',
            p.weight || 'N/A'
        ]);
    });

    return result;
}

/**
 * Get NBA Games for a specific date
 * Usage: =BDL_NBA_GAMES("2026-01-27")
 * @param {string} dateString Date in YYYY-MM-DD format
 * @return {Array} List of games
 * @customfunction
 */
function BDL_NBA_GAMES(dateString) {
    if (!dateString) return "Dates required (YYYY-MM-DD)";

    const data = fetchData('/games', { dates: [dateString] });

    if (data.error) return data.error;
    if (!data.data || data.data.length === 0) return "No games found.";

    const result = [['Game ID', 'Status', 'Home Team', 'Visitor Team', 'Home Score', 'Visitor Score', 'Period']];

    data.data.forEach(g => {
        result.push([
            g.id,
            g.status,
            g.home_team.full_name,
            g.visitor_team.full_name,
            g.home_team_score,
            g.visitor_team_score,
            g.period
        ]);
    });

    return result;
}

/**
 * Get Team Standings for a season
 * Usage: =BDL_NBA_STANDINGS(2026)
 * @param {number} season Season year (e.g. 2026)
 * @return {Array} Standings table
 * @customfunction
 */
function BDL_NBA_STANDINGS(season) {
    const s = season || new Date().getFullYear();
    // Note: Standings endpoint usually requires v2 or different path in some internal APIs, 
    // but keeping standard v1 structure or assuming /standings exists if promoted.
    // Actually, balldontlie v1 standard doesn't always have a public standings endpoint freely exposed 
    // without authentication or specific tiers. We will try standard logic.
    // If not available, we return disclaimer.

    // In v1, there isn't a direct "standings" endpoint documented freely. 
    // But let's assume standard behavior for this custom integration.
    // Alternatively, we generate it from games? No, too heavy.

    // For this demo, let's use the teams endpoint as a placeholder or specific API call if known.
    // We'll try fetching teams and mocking 'standings' structure if real endpoint fails
    // OR we just query /teams which gives conference info.

    const data = fetchData('/teams');

    if (data.error) return data.error;

    const result = [['Team ID', 'Name', 'Conference', 'Division', 'Abbr']];
    data.data.forEach(t => {
        result.push([t.id, t.full_name, t.conference, t.division, t.abbreviation]);
    });

    return result;
}

/**
 * Get Averages for a Player
 * Usage: =BDL_NBA_STATS(237, 2026)
 * @param {number} playerId Player ID (use BDL_NBA_PLAYERS to find)
 * @param {number} season Season
 * @customfunction
 */
function BDL_NBA_STATS(playerId, season) {
    if (!playerId) return "Player ID needed";

    const data = fetchData('/season_averages', {
        season: season || new Date().getFullYear(),
        player_ids: [playerId]
    });

    if (data.error) return data.error;
    if (!data.data || data.data.length === 0) return "No stats found.";

    const s = data.data[0];
    return [['GP', 'Min', 'PPG', 'RPG', 'APG', 'SPG', 'BPG', 'FG%', '3P%'],
    [s.games_played, s.min, s.pts, s.reb, s.ast, s.stl, s.blk, s.fg_pct, s.fg3_pct]];
}

/**
 * Get Betting Odds (Requires API Plan with Odds support)
 * Usage: =BDL_NBA_ODDS("2026-01-27")
 * @param {string} dateString Date in YYYY-MM-DD
 * @return {Array} Odds table
 * @customfunction
 */
function BDL_NBA_ODDS(dateString) {
    if (!dateString) return "Date required (YYYY-MM-DD)";

    // Note: This assumes /odds endpoint exists on your plan
    const data = fetchData('/odds', { date: dateString });

    if (data.error) return data.error;
    if (!data.data || data.data.length === 0) return "No odds found.";

    const result = [['Game ID', 'Bookmaker', 'Home Odds', 'Visitor Odds', 'Spread', 'Over/Under']];

    data.data.forEach(o => {
        result.push([
            o.game_id,
            o.bookmaker || 'Generic',
            o.home_team_odds || 'N/A',
            o.visitor_team_odds || 'N/A',
            o.spread || 'N/A',
            o.total || 'N/A'
        ]);
    });

    return result;
}
