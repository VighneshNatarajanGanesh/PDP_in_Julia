module ParDepPlot

function PDP(mach::Machine, df::DataFrame, targetColumn::String, targets::Array{String})
        #=
        This function returns a 1-D PDP plot of multi-class-classification model

        Arguments:
        mach: The machine (the wrapping the model in data)
        df: The dataframe containing the test data
        targetColumn: The feature for which PDP plot must be drawn
        targets: The different classes of the targetColumn for which plot must be generated

        Returns:
        a plot with separate subplots for each class mentioned in targets argument
        =#

        # cross product the target feature for PDP along with non-target features to get data
        dfCross = crossjoin( DataFrame(targetColumn=df[!, targetColumn]), df[!, filter(x->x!=targetColumn, names(df))] )

        # rename the column from the symbol targetColumn to string in the variable targetColumn
        colnames = pushfirst!(setdiff(names(dfCross), ["targetColumn"]), targetColumn)
        rename!(dfCross, Symbol.(colnames))

        # make the predictions
        yhat = predict(mach, dfCross);
        for x in targets
                dfCross[!, x] = broadcast(pdf, yhat, x)
        end

        # group by the targetColumn
        groupedDfCross = groupby(dfCross, targetColumn)

        #initialize the plot
        p=plot(layout=length(targets))

        # generate the statistics for every class in target column and make the subplot
        for i in 1:length(targets)
                meanYHat = combine(groupedDfCross,targets[i]=>mean=>"mean")
                sort!(meanYHat, targetColumn) # to make sure the the db is in ascending order of target var
                plot!(p, meanYHat[!, targetColumn], meanYHat[!, "mean"], subplot=i, title=targets[i], label="")
        end

        # return the subplot
        return p

end

end # for the module
