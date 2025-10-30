function load_data(data_folder::String, thread_Id::String)
    data_full_raw = readdlm(string(data_folder, "thread_id__", thread_Id, "__data.csv"), ',')
    maturities_raw = vec(readdlm(string(data_folder, "thread_id__", thread_Id, "__maturities.csv"), ','))
    return data_full_raw, maturities_raw
end

function extend_data(data::AbstractMatrix, extension_horizon::Int)
    # extend the data with NaN 
    n_rows, n_cols = size(data)
    extended_data = Array{eltype(data)}(undef, n_rows, n_cols + extension_horizon)
    extended_data[:, 1:n_cols] .= data
    extended_data[:, n_cols+1:end] .= NaN
    return extended_data
end