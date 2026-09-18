"""Physical thresholds in mm/day; events include the threshold itself (>=)."""
WET_THRESHOLD = 1.0
EVENT_THRESHOLDS = (1.0, 10.0, 20.0)
SEASONS = ('ALL', 'DJF', 'MAM', 'JJA', 'SON')


def season_of(date):
    month = int(date[4:6])  # Dates are validated as YYYYMMDD by common_io.
    return ('DJF', 'MAM', 'JJA', 'SON')[(month % 12) // 3]
