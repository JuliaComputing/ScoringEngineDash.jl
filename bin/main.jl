using Dash, PlotlyBase
using HTTP
using CSV, JSON3, JSONTables, DataFrames
using Random
using Statistics: mean, std
using StatsBase: sample
using ShapML
using Loess

const url_base = "http://localhost:8008/api/v1"
# const url_base = get(ENV, "SCORING_URL", "$(url_base)/flux")

@info "dir ::" readdir("/home/ubuntu/.julia/packages/ScoringEngineDash")
const df_tot =  begin
    df_tot = CSV.read(joinpath("/home/ubuntu/.julia/packages/ScoringEngineDash/*/data", "training_data.csv"), DataFrame)
    ## df_tot = CSV.read(joinpath(@__DIR__, "../data", "training_data.csv"), DataFrame)
    dropmissing!(df_tot)
end

const sample_size = 20

# features to be passed to shap function
const features_importance = ["pol_no_claims_discount", "pol_coverage", "pol_duration",
    "pol_sit_duration", "vh_value", "vh_weight", "vh_age", "population",
    "town_surface_area", "drv_sex1", "drv_age1", "pol_pay_freq"]

# available features for one-way effect - only numeric features ATM
const features_effect = ["pol_no_claims_discount", "pol_duration", "pol_sit_duration", "vh_value",
    "vh_weight", "vh_age", "population", "town_surface_area", "drv_age1"]

function get_scores(df::DataFrame, model)
    body = JSON3.write(arraytable(df))
    endpoint = "$(url_base)/$(model)"
    req = HTTP.request("POST", endpoint, [], body)
    res = JSON3.read(req.body, Dict)
    scores = Float64.(res["score"])
    return scores
end

function add_scores!(df::DataFrame)
    scores_flux = get_scores(df, "flux")
    scores_gbt = get_scores(df, "gbt")
    df[:, :flux] .= scores_flux
    df[:, :gbt] .= scores_gbt
    return nothing
end

function pred_shap(model, df)
    scores = get_scores(df::DataFrame, model)
    pred_df = DataFrame(score = scores)
    return pred_df
end

function run_shap(df, model = "flux")
    data_shap = ShapML.shap(
        explain = copy(df),
        reference = copy(df),
        target_features = features_importance,
        model = model,
        predict_function = pred_shap,
        sample_size = sample_size,
        seed = 123)
    return data_shap
end

function plot_shap(data_shap, feat = "vh_age")
    df = data_shap[data_shap.feature_name.==feat, :]
    transform!(df, :feature_value => ByRow(x -> convert(Float64, x)) => :feature_value)
    model = loess(df[:, :feature_value], df[:, :shap_effect], span = 1.0)
    smooth_x = range(extrema(df[:, :feature_value])...; length = 10)
    smooth_y = Loess.predict(model, smooth_x)
    return (df = df, smooth_x = smooth_x, smooth_y = smooth_y)
end

function get_feat_importance(data_shap)
    dfg = groupby(data_shap, :feature_name)
    df = combine(dfg, :shap_effect => (x -> mean(abs.(x))) => :shap_effect)
    sort!(df, :shap_effect, rev = false)
    return df
end

years = unique(df_tot[!, "year"])
rng = Random.MersenneTwister(123)

app = dash(external_stylesheets = ["https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css"])
# app = dash()

app.layout = html_div(className = "p-5") do
    html_h3("Scoring Engine - Model Exploration Dashboard"),
    html_div([
            html_p("Select Feature"),
            html_div(dcc_dropdown(
                id = "xaxis-column",
                options = [(label = i, value = i) for i in features_effect],
                value = "vh_age"), className = "col-md-4"),
            html_div(html_button("Update Sample!", id = "btn-1"), className = "py-3")
        ], className = "col-md-12"),
    html_div(
        dcc_graph(id = "plot1"); className = "col-md-12 p-3"
    ), html_div(children = [
            html_div(dcc_graph(id = "plot2"); className = "col-md-6 p-3")
            html_div(dcc_graph(id = "plot3"); className = "col-md-6 p-3")
        ], className = "row")
end

callback!(
    app,
    Output("plot1", "figure"),
    Output("plot2", "figure"),
    Output("plot3", "figure"),
    Input("btn-1", "n_clicks"),
    Input("xaxis-column", "value"),
) do btn_n_clicks, xaxis_column_name
    ids = sample(rng, 1:nrow(df_tot), 20, replace = false, ordered = true)
    df = df_tot[ids, :]
    add_scores!(df)
    data_flux = run_shap(df, "flux")
    feat_flux = get_feat_importance(data_flux)
    shap_flux = plot_shap(data_flux, xaxis_column_name)

    data_gbt = run_shap(df, "gbt")
    feat_gbt = get_feat_importance(data_gbt)
    shap_gbt = plot_shap(data_gbt, xaxis_column_name)

    # println("ids: ", ids)

    return (Plot(
            [scatter(x = shap_flux[:df][:, :feature_value], y = shap_flux[:df][:, :shap_effect], mode = "markers", marker = attr(color = "red", opacity = 0.5, size = 12), name="flux"),
                scatter(x = shap_flux[:smooth_x], y = shap_flux[:smooth_y], mode = "lines", marker = attr(color = "purple", size = 12), name="flux"),
                scatter(x = shap_gbt[:df][:, :feature_value], y = shap_gbt[:df][:, :shap_effect], mode = "markers", marker = attr(color = "green", opacity = 0.5, size = 12), name="gbt"),
                scatter(x = shap_gbt[:smooth_x], y = shap_gbt[:smooth_y], mode = "lines", marker = attr(color = "blue", size = 12), name="gbt")],
            Layout(
                title = "Flux vs GBT $xaxis_column_name effects",
                plot_bgcolor = "white",
                paper_bgcolor = nothing,
                xaxis = attr(
                    title = xaxis_column_name,
                    showgrid = true,
                    gridcolor = "lightgray",
                    showline = false,
                    linecolor = "black",
                    titlefont_color = "black"),
                yaxis = attr(
                    title = "SHAP effect",
                    showgrid = true,
                    gridcolor = "lightgray",
                    showline = false,
                    linecolor = "black",
                    titlefont_color = "black")
            )
        ),
        Plot(
            bar(y = feat_flux[:, :feature_name], x = feat_flux[:, :shap_effect], orientation = "h", marker = attr(color = "red", opacity = 0.5)),
            Layout(
                title = "Flux geature importance",
                plot_bgcolor = "white",
                paper_bgcolor = nothing,
                xaxis = attr(
                    title = "xaxis title",
                    showgrid = true,
                    gridcolor = "lightgray",
                    showline = true,
                    linecolor = "black",
                    titlefont_color = "black"),
                yaxis = attr(
                    title = "yaxis title",
                    showgrid = true,
                    gridcolor = "lightgray",
                    showline = true,
                    linecolor = "black",
                    titlefont_color = "black")
            )
        ),
        Plot(
            bar(y = feat_gbt[:, :feature_name], x = feat_gbt[:, :shap_effect], orientation = "h", marker = attr(color = "green", opacity = 0.5)),
            Layout(
                title = "GBT feature importance",
                plot_bgcolor = "white",
                paper_bgcolor = nothing,
                xaxis = attr(
                    title = "xaxis title",
                    showgrid = true,
                    gridcolor = "lightgray",
                    showline = true,
                    linecolor = "black",
                    titlefont_color = "black"),
                yaxis = attr(
                    title = "yaxis title",
                    showgrid = true,
                    gridcolor = "lightgray",
                    showline = true,
                    linecolor = "black",
                    titlefont_color = "black")
            )
        ))
end

# run_server(app, "127.0.0.1", 80, debug = true)
run_server(app, "0.0.0.0", 80, debug = true)
