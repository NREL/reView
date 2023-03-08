"""reView default layout options."""
# pylint: disable=import-error
from itertools import chain

import us  # noqa: E0401

from reView.utils.constants import COLORS, COLORS_Q

REGIONS = {
    "Pacific": ["Oregon", "Washington"],
    "Mountain": ["Colorado", "Idaho", "Montana", "Wyoming"],
    "Great Plains": [
        "Iowa",
        "Kansas",
        "Missouri",
        "Minnesota",
        "Nebraska",
        "North Dakota",
        "South Dakota",
    ],
    "Great Lakes": ["Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin"],
    "Northeast": [
        "Connecticut",
        "New Jersey",
        "New York",
        "Maine",
        "New Hampshire",
        "Massachusetts",
        "Pennsylvania",
        "Rhode Island",
        "Vermont",
    ],
    "California": ["California"],
    "Southwest": ["Arizona", "Nevada", "New Mexico", "Utah"],
    "South Central": ["Arkansas", "Louisiana", "Oklahoma", "Texas"],
    "Southeast": [
        "Alabama",
        "Delaware",
        "District of Columbia",
        "Florida",
        "Georgia",
        "Kentucky",
        "Maryland",
        "Mississippi",
        "North Carolina",
        "South Carolina",
        "Tennessee",
        "Virginia",
        "West Virginia",
    ]
}


BASEMAP_OPTIONS = [
    {"label": "Light", "value": "light"},
    {"label": "Dark", "value": "dark"},
    {"label": "Basic", "value": "basic"},
    {"label": "Outdoors", "value": "outdoors"},
    {"label": "Satellite", "value": "satellite"},
    {"label": "Satellite Streets", "value": "satellite-streets"},
]
CHART_OPTIONS = [
    {"label": "Cumulative Capacity", "value": "cumsum"},
    {"label": "Bivariate - Scatterplot", "value": "scatter"},
    {"label": "Bivariate - Binned Line Plot", "value": "binned"},
    {"label": "Histogram", "value": "histogram"},
    {"label": "Boxplot", "value": "box"},
    {"label": "Characterizations", "value": "char_histogram"},
]
COLOR_OPTIONS = [{"label": k, "value": k} for k, _ in COLORS.items()]
COLOR_Q_OPTIONS = [{"label": k, "value": k} for k, _ in COLORS_Q.items()]
REGION_OPTIONS = [{"label": k, "value": k} for k in REGIONS.keys()]

# pylint: disable=no-member
STATE_OPTIONS = [{"label": s.name, "value": s.name} for s in us.STATES] + [
    {"label": "Onshore", "value": "onshore"},
    {"label": "Offshore", "value": "offshore"},
]
