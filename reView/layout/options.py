"""reView default layout options."""
# pylint: disable=import-error
import us  # noqa: E0401

from reView.utils.constants import COLORS, COLORS_Q


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
REGION_OPTIONS = [
    {"label": "Pacific", "value": "Pacific"},
    {"label": "Mountain", "value": "Mountain"},
    {"label": "Great Plains", "value": "Great Plains"},
    {"label": "Great Lakes", "value": "Great Lakes"},
    {"label": "Northeast", "value": "Northeast"},
    {"label": "California", "value": "California"},
    {"label": "Southwest", "value": "Southwest"},
    {"label": "South Central", "value": "South Central"},
    {"label": "Southeast", "value": "Southeast"}
]

# pylint: disable=no-member
STATE_OPTIONS = [{"label": s.name, "value": s.name} for s in us.STATES] + [
    {"label": "Onshore", "value": "onshore"},
    {"label": "Offshore", "value": "offshore"},
]
