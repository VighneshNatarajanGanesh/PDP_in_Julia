function PDP(mach::Machine, df::DataFrame, targetColumn::String, targets::Array{String})

        # cross the target feature for PDP along with non-target features to get data
        dfCross = crossjoin( DataFrame(targetColumn=df[!, targetColumn]), df[!, filter(x->x!=targetColumn, names(df))] )

        #rename the column from thr symbol targetColumn to string in target column
        colnames = pushfirst!(setdiff(names(dfCross), ["targetColumn"]), targetColumn)
        rename!(dfCross, Symbol.(colnames))

        # make the predictions
        yhat = predict(mach, dfCross);
        for x in targets
                dfCross[!, x] = broadcast(pdf, yhat, x)
        end
        groupedDfCross = groupby(dfCross, targetColumn)

        p=plot(layout=length(targets))
        for i in 1:length(targets)
                meanYHat = combine(groupedDfCross,targets[i]=>mean=>"mean")
                sort!(meanYHat, targetColumn) # to make sure the the db is in ascending order of target var
                plot!(p, meanYHat[!, targetColumn], meanYHat[!, "mean"], subplot=i, title=targets[i], label="")
        end

        return p

end
