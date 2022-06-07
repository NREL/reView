"""reView style specifications."""
import copy


# Move to CSS
BUTTON_STYLES = {
    "on": {
        "height": "35px",
        "width": "175px",
        "padding": "0px",
        "background-color": "#FCCD34",
        "border-radius": "4px",
        "border-color": "#1663b5",
        "font-family": "Times New Roman",
        "font-size": "12px",
        "margin-top": "-2px",
    },
    "off": {
        "height": "5px",
        "width": "175px",
        "text-align": "center",
        "padding": "0px",
        "border-color": "#1663b5",
        "background-color": "#b89627",
        "border-radius": "4px",
        "font-family": "Times New Roman",
        "font-size": "12px",
        "margin-top": "-2px",
    },
    "navbar": {
        "height": "55px",
        "width": "190px",
        "background-color": "#FCCD34",
        "border-radius": "4px",
        "border-color": "#1663b5",
        "font-family": "Times New Roman",
        "font-size": "18px",
        "textTransform": "none"
    },
}
BOTTOM_DIV_STYLE = {
    "display": "table",
    "width": "100%",
}
TAB_STYLE = {"height": "25px", "padding": "0"}
TABLET_STYLE = {"line-height": "25px", "padding": "0"}

# Everything below goes into a css
TABLET_STYLE_CLOSED = {
    **TABLET_STYLE,
    **{"border-bottom": "1px solid #d6d6d6"},
}
TAB_BOTTOM_SELECTED_STYLE = {
    "borderBottom": "1px solid #1975FA",
    "borderTop": "1px solid #d6d6d6",
    "line-height": "25px",
    "padding": "0px",
}
RC_STYLES = copy.deepcopy(BUTTON_STYLES)
RC_STYLES["off"]["border-color"] = RC_STYLES["on"]["border-color"] = "#1663b5"
RC_STYLES["off"]["border-width"] = RC_STYLES["on"]["border-width"] = "3px"
RC_STYLES["off"]["margin-top"] = RC_STYLES["on"]["margin-top"] = "-3px"
RC_STYLES["off"]["display"] = RC_STYLES["on"]["display"] = "table:cell"
RC_STYLES["off"]["float"] = RC_STYLES["on"]["float"] = "right"
# Everything above goes into css


OPTION_TITLE_STYLE = {
    "float": "left",
    "font-size": "12px",
    "font-family": "Helvetica",
    "width": "100%",
    "margin-bottom": "-1px"
}
OPTION_STYLE = {
    "height": "30px",
    "width": "100%",
    "text-align": "center",
    "display": "inline-block",
    "font-size": "12px",
    "margin-top": "-1px"
}
