# -*- coding: utf-8 -*-
"""WIP PCA plot stuff."""

# PCA_DF = pd.read_csv(
#     Path.home() / "review_datasets" / "hydrogen_pca" / "pca_df_300_sites.csv"
# )


# def build_pca_plot(
#     color, x, y, z, camera=None, ymin=None, ymax=None, state="CONUS"
# ):
#     """Build a Plotly pca plot."""

#     # Create hover text
#     # if units == "category":
#     #     df["text"] = (
#     #         df["county"]
#     #         + " County, "
#     #         + df["state"]
#     #         + ": <br>   "
#     #         + df[y].astype(str)
#     #         + " "
#     #         + units
#     #     )
#     # else:
#     #     extra_str = ""
#     #     if "hydrogen_annual_kg" in df:
#     #         extra_str += (
#     #             "<br>    H2 Supply:    "
#     #             + df["hydrogen_annual_kg"].apply(lambda x: f"{x:,}")
#     #             + " kg    "
#     #         )
#     #     if "dist_to_selected_load" in df:
#     #         extra_str += (
#     #             "<br>    Dist to load:    "
#     #             + df["dist_to_selected_load"].apply(lambda x: f"{x:,.2f}")
#     #             + " km    "
#     #         )

#     #     df["text"] = (
#     #         df["county"]
#     #         + " County, "
#     #         + df["state"]
#     #         + ":"
#     #         + extra_str
#     #         + f"<br>    {convert_to_title(y)}:   "
#     #         + df[y].round(2).astype(str)
#     #         + " "
#     #         + units
#     #     )

#     # marker = dict(
#     #     color=df[y],
#     #     colorscale=pcolor,
#     #     cmax=None if ymax is None else float(ymax),
#     #     cmin=None if ymin is None else float(ymin),
#     #     opacity=1.0,
#     #     reversescale=rev_color,
#     #     size=point_size,
#     #     colorbar=dict(
#     #         title=dict(
#     #             text=units,
#     #             font=dict(
#     #                 size=15, color="white", family="New Times Roman"
#     #             ),
#     #         ),
#     #         tickfont=dict(color="white", family="New Times Roman"),
#     #     ),
#     # )

#     # Create data object
#     # figure = px.scatter_mapbox(
#     #     data_frame=df,
#     #     lon="longitude",
#     #     lat="latitude",
#     #     custom_data=["sc_point_gid", "print_capacity"],
#     #     hover_name="text",
#     # )

#     principal_df = PCA_DF[PCA_DF.State == state]
#     features = [
#         "electrolyzer_size_ratio",
#         "wind_cost_multiplier",
#         "fcr",
#         "water_cost_multiplier",
#         "pipeline_cost_multiplier",
#         "electrolyzer_size_mw",
#         "electrolyzer_capex_per_mw",
#     ]
#     range_color = (
#         None if ymin is None else float(ymin),
#         None if ymax is None else float(ymax),
#     )
#     figure = px.scatter_3d(
#         principal_df,
#         x=x,
#         y=y,
#         z=z,
#         color=color,
#         range_color=range_color,
#         size_max=15,
#         # marker=dict(size=3, symbol="circle"),
#         hover_name=principal_df[color],
#         hover_data=features,
#         custom_data=["file"],
#         # text=[f for f in principal_df['file']]
#     )
#     # figure.update_traces(marker=marker)
#     if camera is not None:
#         figure.update_layout(scene_camera=camera)
#     # figure = make_subplots(rows=1, cols=2,
#     #                        shared_xaxes=True,
#     #                        shared_yaxes=True,
#     #                        specs=[[
#     #                            {'type': 'surface'},
#     #                            {'type': 'surface'}
#     #                         ]],
#     #                     # vertical_spacing=0.02
#     #                     )
#     # scatter = go.Scatter3d(x = principal_df['pc1'],
#     #                        y = principal_df['pc2'],
#     #                        z = principal_df['pc3'],
#     #                        mode ='markers',
#     #                        marker = dict(
#     #                         size = 12,
#     #                         color = principal_df[y],
#     #                         # colorscale ='Viridis',
#     #                         # opacity = 0.8
#     #                     )
#     #                     )
#     # scatter2 = go.Scatter3d(x = principal_df['pc1'],
#     #                        y = principal_df['pc2'],
#     #                        z = principal_df['pc3'],
#     #                        mode ='markers',
#     #                        marker = dict(
#     #                         size = 12,
#     #                         color = principal_df[y],
#     #                         # colorscale ='Viridis',
#     #                         # opacity = 0.8
#     #                     )
#     #                     )

#     # figure.add_trace(scatter, row=1, col=1)
#     # figure.add_trace(scatter2, row=1, col=2)

#     # figure.update_layout(height=600, width=600,
#     #                      title_text="Stacked Subplots with Shared X-Axes")

#     # Update the layout
#     # layout_ = build_map_layout(
#     #     title, basemap, showlegend, ymin, ymax
#     # )
#     # figure.update_layout(**layout_)

#     return figure



# @app.callback(
#     Output("pca_plot_value_1", "options"),
#     Output("pca_plot_value_1", "value"),
#     Output("pca_plot_value_2", "options"),
#     Output("pca_plot_value_2", "value"),
#     Output("pca_plot_axis1", "options"),
#     Output("pca_plot_axis1", "value"),
#     Output("pca_plot_axis2", "options"),
#     Output("pca_plot_axis2", "value"),
#     Output("pca_plot_axis3", "options"),
#     Output("pca_plot_axis3", "value"),
#     Output("pca_plot_region", "options"),
#     Output("pca_plot_region", "value"),
#     # Input("project", "value"),
#     Input("pca_scenarios", "style"),
# )
# def set_pca_variable_options(
#     # project,
#     pca_scenarios_style,
# ):
#     logger.debug("Setting variable target options")
#     # config = Config(project)
#     variable_options = [{"label": "None", "value": "None"}]
#     axis_options = [
#         {"label": "pc1", "value": "pc1"},
#         {"label": "pc2", "value": "pc2"},
#         {"label": "pc3", "value": "pc3"},
#     ]
#     region_options = [{"label": "CONUS", "value": "CONUS"}]
#     is_showing = (
#         pca_scenarios_style and pca_scenarios_style.get("display") != "none"
#     )
#     if is_showing:
#         variable_options += [
#             {"label": convert_to_title(col), "value": col}
#             for col in PCA_DF.columns
#             if col not in {"pc1", "pc2", "pc3", "file", "State"}
#         ]
#         axis_options += [
#             {"label": convert_to_title(col), "value": col}
#             for col in PCA_DF.columns
#             if col not in {"pc1", "pc2", "pc3", "file", "State"}
#         ]
#         region_options += [
#             {"label": convert_to_title(state), "value": state}
#             for state in PCA_DF.State.unique()
#             if state != "CONUS"
#         ]
#     return (
#         variable_options,
#         variable_options[-1]["value"],
#         variable_options,
#         variable_options[-1]["value"],
#         axis_options,
#         "pc1",
#         axis_options,
#         "pc2",
#         axis_options,
#         "pc3",
#         region_options,
#         "CONUS",
#     )


# @app.callback(
#     Output("pca_plot_map_value", "options"),
#     Output("pca_plot_map_value", "value"),
#     Input("project", "value"),
# )
# def set_pca_plot_options(project):
#     """"""
#     logger.debug("Setting pca plot options")
#     config = Config(project)
#     # TODO: Remove hardcoded path
#     # path = choose_scenario(scenario_options, config)
#     path = ("C:\\Users\\ppinchuk\\review_datasets\\hydrogen\\review_pca\\"
            # "wind_flat_esr01_wcm0_ecpm0_f0035_wcm10_pcm05_nrwal_00.csv"
#     plot_options = [{"label": "Variable", "value": "Variable"}]
#     if path and os.path.exists(path):
#         data = pd.read_csv(path)
#         columns = [c for c in data.columns if c.lower() not in SKIP_VARS]
#         titles = {col: convert_to_title(col) for col in columns}
#         titles.update(config.titles)
#         if titles:
#             for k, v in titles.items():
#                 plot_options.append({"label": v, "value": k})

#     return plot_options, plot_options[-1]["value"]


# @app.callback(
#     [
#         Output("pca_plot_1", "figure"),
#         Output("pca_plot_2", "figure"),
#         # Output("mapcap", "children"),
#         # Output("pca_plot_1", "clickData"),
#     ],
#     [
#         Input("pca_scenarios", "style"),
#         Input("pca_plot_value_1", "value"),
#         Input("pca_plot_value_2", "value"),
#         Input("pca_plot_1", "relayoutData"),
#         Input("pca_plot_2", "relayoutData"),
#         # Input("map_signal", "children"),
#         # Input("basemap_options", "value"),
#         # Input("color_options", "value"),
#         # Input("chart", "selectedData"),
#         # Input("map_point_size", "value"),
#         # Input("rev_color", "n_clicks"),
#         Input("pca1_color_min", "value"),
#         Input("pca1_color_max", "value"),
#         Input("pca2_color_min", "value"),
#         Input("pca2_color_max", "value"),
#         Input("pca_plot_axis1", "value"),
#         Input("pca_plot_axis2", "value"),
#         Input("pca_plot_axis3", "value"),
#         Input("pca_plot_region", "value"),
#         # Input("pca_plot_1", "clickData"),
#     ],
#     # [
#     # State("project", "value"),
#     # State("map", "relayoutData"),
#     # State("map_function", "value"),
#     # ],
# )
# def make_pca_plot(
#     pca_scenarios_style,
#     pca_plot_value_1,
#     pca_plot_value_2,
#     data_plot_one,
#     data_plot_two,
#     uymin1,
#     uymax1,
#     uymin2,
#     uymax2,
#     pca_plot_axis1,
#     pca_plot_axis2,
#     pca_plot_axis3,
#     pca_plot_region
#     # clicksel
# ):
#     """Make the pca plot."""

#     # Don't update if selected data triggered this?
#     logger.debug("PCA TRIGGER: %s", trig)
#     if pca_scenarios_style.get("display") == "none":
#         raise PreventUpdate  # @IgnoreException

#     if pca_plot_value_1 == "None" or pca_plot_value_2 == "None":
#         raise PreventUpdate  # @IgnoreException

#     # print(clicksel)
#     # if clicksel and clicksel.get('points'):
#     #     raise PreventUpdate

#     # df, demand_data = apply_all_selections(
#     #     df, map_project, chartsel, mapsel, clicksel
#     # )

#     logger.debug("Building pca plot")
#     if trig == "pca_plot_1.relayoutData" and data_plot_one:
#         camera = data_plot_one.get("scene.camera")
#     elif trig == "pca_plot_2.relayoutData" and data_plot_two:
#         camera = data_plot_two.get("scene.camera")
#     else:
#         camera = None

#     figure = build_pca_plot(
#         pca_plot_value_1,
#         pca_plot_axis1,
#         pca_plot_axis2,
#         pca_plot_axis3,
#         camera,
#         uymin1,
#         uymax1,
#         pca_plot_region,
#     )
#     figure2 = build_pca_plot(
#         pca_plot_value_2,
#         pca_plot_axis1,
#         pca_plot_axis2,
#         pca_plot_axis3,
#         camera,
#         uymin2,
#         uymax2,
#         pca_plot_region,
#     )

#     return figure, figure2  # , None
