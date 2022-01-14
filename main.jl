using Dash, PlotlyJS
using HTTP
using CSV, JSON3, JSONTables, DataFrames
using Statistics: mean, std
using StatsBase: sample
using ShapML

df_tot = CSV.read(joinpath(@__DIR__, "data", "training_data.csv"), DataFrame)

df = df_tot[1:1000, :];
body = JSON3.write(arraytable(df));
JSON3.read(body) |> jsontable |> DataFrame;

function get_scores(df::DataFrame)
    body = JSON3.write(arraytable(df))
    req = HTTP.request("POST", "http://localhost:8008/api/v1/risk", [], body)
    res = JSON3.read(req.body, Dict)
    flux = Float64.(res["score_flux"])
    gbt = Float64.(res["score_gbt"])
    return (flux = flux, gbt = gbt)
end

function add_scores!(df::DataFrame)
    scores = get_scores(df)
    df[:, :flux] .= scores[:flux]
    df[:, :gbt] .= scores[:gbt]
    return nothing
end


function pred_shap_flux(model, df)
    pred = get_scores(df::DataFrame)
    pred_df = DataFrame(score = pred[:flux])
    return pred_df
end

sample_size = 30
explain, reference = copy(df[1:10, :]), copy(df[1:10, :])

features = ["pol_no_claims_discount", "pol_coverage", "pol_duration", "pol_sit_duration", "vh_value", "vh_weight", "vh_age", "population", "town_surface_area", "drv_sex1", "drv_age1", "pol_pay_freq"]
@time data_shap = ShapML.shap(
    explain = explain,
    reference = reference,
    target_features = features,
    model = "flux",
    predict_function = pred_shap_flux,
    sample_size = sample_size,
    seed = 123
)
data_shap[data_shap.feature_name.=="drv_age1", :]
data_shap[data_shap.feature_name.=="vh_age", :]

function run_shap(df, model = "flux")
    data_shap = ShapML.shap(
        explain = copy(df),
        reference = copy(df),
        target_features = features,
        model = "flux",
        predict_function = pred_shap_flux,
        sample_size = sample_size,
        seed = 123)
    return data_shap
end

function plot_shap(data_shap, feat)
    df = data_shap[data_shap.feature_name.=="vh_age", :]
end

function get_feat_importance(data_shap)
    dfg = groupby(data_shap, :feature_name)
    df = combine(dfg, :shap_effect => (x -> mean(abs.(x))) => :shap_effect)
    sort!(df, :shap_effect, rev = false)
    return df
end

available_indicators = unique(df_tot[!, "pol_coverage"])
years = unique(df_tot[!, "year"])
rng = Random.MersenneTwister(123)

app = dash(external_stylesheets = ["https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css"])

app.layout = html_div(className = "p-5") do
    html_div([
            html_p("Select Feature"),
            html_div(dcc_dropdown(
                    id = "xaxis-column",
                    options = [(label = i, value = i) for i in available_indicators],
                    value = "Fertility rate, total (births per woman)")),
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
    Input("xaxis-column", "value"),
    Input("btn-1", "n_clicks"),
) do xaxis_column_name, btn_n_clicks
    ids = sample(rng, 1:nrow(df_tot), 20, replace = false, ordered = true)
    df = df_tot[ids, :]
    add_scores!(df)
    data_shap = run_shap(df)
    feat_imp = get_feat_importance(data_shap)
    println("ids: ", ids)
    return (Plot(
            # [scatter(x = rand(100), y = rand(100), mode = "markers", marker = attr(color="darkgreen"))]
            scatter(x = df[:, :flux], y = df[:, :gbt], mode = "markers", marker = attr(color = "darkgreen", size = 12)),
            Layout(
                title = "Flux vs GBT predictions",
                plot_bgcolor = "white",
                paper_bgcolor = nothing,
                xaxis = attr(
                    title = "xaxis title",
                    showgrid = true,
                    gridcolor = "lightgray",
                    showline = false,
                    linecolor = "black",
                    titlefont_color = "black"),
                yaxis = attr(
                    title = "yaxis title",
                    showgrid = true,
                    gridcolor = "lightgray",
                    showline = false,
                    linecolor = "black",
                    titlefont_color = "black")
            )
        ),
        Plot(
            bar(y = feat_imp[:, :feature_name], x = feat_imp[:, :shap_effect], orientation = "h", marker = attr(color = "red", opacity = 0.5)),
            Layout(
                title = "Flux Feature Importance",
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
            scatter(x = df[:, :flux], y = df[:, :gbt], mode = "markers", marker = attr(color = "purple", size = 12)),
            Layout(
                title = "Flux vs GBT predictions",
                plot_bgcolor = "white",
                paper_bgcolor = nothing,
                xaxis = attr(
                    title = "xaxis title",
                    showgrid = true,
                    gridcolor = "lightgray",
                    showline = false,
                    linecolor = "black",
                    titlefont_color = "black"),
                yaxis = attr(
                    title = "yaxis title",
                    showgrid = true,
                    gridcolor = "lightgray",
                    showline = false,
                    linecolor = "black",
                    titlefont_color = "black")
            )
        ))
end

run_server(app, "127.0.0.1", 80, debug = true)