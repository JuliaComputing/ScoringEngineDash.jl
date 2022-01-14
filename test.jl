using HTTP
using CSV, JSON3, JSONTables, DataFrames

df_tot = CSV.read(joinpath(@__DIR__, "data", "training_data.csv"), DataFrame)
df = df_tot[1:2, :]
body = JSON3.write(arraytable(df))
JSON3.read(body) |> jsontable |> DataFrame

req = HTTP.request("GET", "http://localhost:8008")
req = HTTP.request("POST", "http://localhost:8008/api/v1/risk", [], body)
