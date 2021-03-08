module ParDepPlot

function PDPCategorical(mach::Machine, df::DataFrame, depColumn::String, categories::Vector{String})
        #=
        This function returns a 1-D PDP plot of multi-class-classification model

        Arguments:
        mach: The machine (the wrapping the model in data)
        df: The dataframe containing the test data
        depColumn: The feature for which PDP plot must be drawn
        categories: The different classes of the targetColumn for which plot must be generated

        Returns:
        a plot with separate subplots for each category mentioned in categories argument
        =#

        # cross product the target feature for PDP along with non-target features to get data
        dfCross = crossjoin( DataFrame(depColumn=df[!, depColumn]), df[!, filter(x->x!=depColumn, names(df))] )

        # rename the column from the symbol targetColumn to string in the variable targetColumn
        colnames = pushfirst!(setdiff(names(dfCross), ["depColumn"]), depColumn)
        rename!(dfCross, Symbol.(colnames))

        # make the predictions
        yhat = predict(mach, dfCross);
        for x in categories
                dfCross[!, x] = broadcast(pdf, yhat, x)
        end

        # group by the depColumn
        groupedDfCross = groupby(dfCross, depColumn)

        #initialize the plot
        p=plot(layout=length(categories))

        # generate the statistics for every class in target column and make the subplot
        for i in 1:length(categories)
                meanYHat = combine(groupedDfCross,categories[i]=>mean=>"mean")
                sort!(meanYHat, depColumn) # to make sure the the db is in ascending order of target var
                plot!(p, meanYHat[!, depColumn], meanYHat[!, "mean"], subplot=i, title=string(categories[i]))
        end

        # return the subplot
        return p

end

end # for the module
