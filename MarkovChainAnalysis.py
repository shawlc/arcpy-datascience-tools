"""
THIS SCRIPT APPLIES CLASSIC MARKOV CHAIN ANALYSIS

To create an ArcToolbox tool with which to execute this script, do the following.
1   In  ArcMap > Catalog > Toolboxes > My Toolboxes, either select an existing toolbox
    or right-click on My Toolboxes and use New > Toolbox to create (then rename) a new one.
2   Drag (or use ArcToolbox > Add Toolbox to add) this toolbox to ArcToolbox.
3   Right-click on the toolbox in ArcToolbox, and use Add > Script to open a dialog box.
4   In this Add Script dialog box, use Label to name the tool being created, and press Next.
5   In a new dialog box, browse to the .py file to be invoked by this tool, and press Next.
6   In the next dialog box, specify the following inputs (using dropdown menus wherever possible)
    before pressing OK or Finish.

        DISPLAY NAME                        DATA TYPE      PROPERTY>DIRECTION>VALUE    PROPERTY>DEFAULT>VALUE
        Input Shapefile                     Shapefile      Input
        Field at Beginning of Time Series   Fields         Input                       Obtained From = Inpute Shapefile
        Field at End of Time Series         Fields         Input                       Obtained From = Inpute Shapefile
        Discrete Intervals                  Long           Input                       Default = 4
        Output Shapefile                    Shapefile      Output

Markov Chain Analysis can be used to predict new values in 3D data by calculating the probability that any discrete
state will transform into any other discrete state, including itself, in the future. This script takes in a shapefile
with an attribute table with some subset of fields that represent a sorted time series. An example attribute table
would have some Fields 1-N where each subsequent field is located to the right of the previous and represents
a single temporal event occuring after the previous. The parameters "Field at Beginning of Time Series" and
"Field at End of Time Series" constrain where the time series fields being and end.

The script also takes in discrete intervals, which are necessary to keep the Markov Chain Analysis efficient. The
number of intervals are used to discretize a matrix of the entire time series by dividing by the maximum value.
The discretized matrix is used to create a Markov transition matrix.

Values from the most recent field are extracted into an array that is iterated over the transition matrix to produce
a matrix where each element is the probability of the most recent value in the same row entering the discrete interval
of the column in the next iteration of the time series. Values in the next future iteration of the time series are
predicted by finding the indices of the maximum probabilities for each column in the array. These indices are multiplied
by the discrete interval to produce the final prediction vector.

The final outputs include a copy of the input shapefile with a discretized field of the most recent field in the time
series, a field of the future predicted values, and a dbf table with the transition matrix.


7   To later revise any of this, right-click to the tool's name and select Properties.
"""

# Import external modules
import sys,string,os,arcpy,traceback
import numpy as np

arcpy.env.overwriteOutput=True

try:

        shapefile=arcpy.GetParameterAsText(0)
        startField=arcpy.GetParameterAsText(1)
        endField=arcpy.GetParameterAsText(2)
        discrete=int(arcpy.GetParameterAsText(3))
        output=arcpy.GetParameterAsText(4)
        outputTable=arcpy.GetParameterAsText(5)

        #CREATE AN ARRAY FROM THE ATTRIBUTE TABLE

        #This function was borrowed from stackexchange
        #https://gis.stackexchange.com/questions/101540/finding-the-index-of-a-field-with-its-name-using-arcpy
        def findindex(table,fieldname):
            return [i.name for i in arcpy.ListFields(table)].index(fieldname)

        #Create indices from the start and end fields of the sorted time series fields
        startindex = findindex(shapefile,startField)
        endindex = findindex(shapefile,endField)

        arcpy.AddMessage("\nBeginning index is "+str(startindex)+" and end index is "+str(endindex))

        #SearchCursor and ListFields are used to extract values from the attribute table
        rowCursor=arcpy.SearchCursor(shapefile)
        rowCount = int(arcpy.GetCount_management(shapefile).getOutput(0))
        fieldList=arcpy.ListFields(shapefile)

        #The following function converts the fields in the sorted time series into a Numpy array
        def attribute_to_array(cursor,field_list,start,end):
            transition_array=np.zeros([rowCount,len(fieldList[startindex:endindex])])

            for rowInd, row in enumerate(cursor):
                # Loops through each of the fields for each row
                for fieldInd, field in enumerate(field_list[start:end]):
                    # Use the field name to get the value
                    transition_array[rowInd,fieldInd]=row.getValue(field.name)

            return transition_array

        transitionArray=attribute_to_array(rowCursor,fieldList,startindex,endindex)

        #Discrete intervals are calculated based on the maximum value in the entire time series
        def discretize(timeseries,intervals):
            series_max=max(map(lambda row:max(row),timeseries))
            arcpy.AddMessage("\nMax value is "+str(series_max))

            interval=np.ceil(1.0*series_max/intervals)
            arcpy.AddMessage("\nInterval is "+str(interval))
            return [[int(np.ceil(1.0*val/interval)) for val in row] for row in timeseries]

        #This function isolates the discrete intervals (This can be rewritten more efficiently)
        def discrete_interval(timeseries, intervals):
            series_max=max(map(lambda row:max(row),timeseries))
            return np.ceil(1.0*series_max/intervals)

        #The transition matrix was inspired from code from stackexchange
        #https://stackoverflow.com/questions/46657221/generating-markov-transition-matrix-in-python
        def generate_transition_matrix(timeseries,intervals):

            timeseries = discretize(timeseries,intervals)
            n=max(map(lambda row:max(row),timeseries))  #number of states

            #An array times a scalar appends the array with itself the number of times as the scalar.
            # This creates an n x n matrix
            transition_matrix=np.zeros([n,n])

            #zip creates a list of multiple lists.
            #The input array is zipped with an offset array to count all transitions
            #The discrete values are subtracted by 1 to convert them into "indices"
            for row in timeseries:
                for (i,j) in zip(row,row[1:]):
                    transition_matrix[i-1][j-1]+=1

            #The transition matrix is converted from counts into conditional probabilities
            #The final transition matrix exists on the principle that an element located in row i and column j
            #represents the probability of a transition from state i to state j
            transition_matrix=1.0*transition_matrix/map(lambda x : 1 if x == 0 else x,
                                                        transition_matrix.sum(axis=-1, keepdims=True))

            arcpy.AddMessage("\nAverage conditional probability is "+
                             str(np.mean(map(lambda row:np.mean(row),transition_matrix))))

            return transition_matrix

        #A copy of the shapefile is created for the Markov Chain predictions to be produced
        arcpy.Copy_management(shapefile,output)

        outCursor=arcpy.UpdateCursor(output)
        global_interval = discrete_interval(transitionArray,discrete)

        intfield = endField+"Int"

        arcpy.Copy_management(shapefile,output)
        arcpy.AddField_management(output,intfield,"DOUBLE",20,5)

        outCursor=arcpy.UpdateCursor(output)

        #This for loop converts the most recent values in the time series into their discrete intervals to be placed
        #In a new field called "TheMostRecentFieldInt"
        for record in outCursor:
            current_value=record.getValue(endField)
            discrete_value=int(np.ceil(1.0*current_value/global_interval))
            record.setValue(intfield,discrete_value)
            outCursor.updateRow(record)

        # Delete row and update cursor objects to avoid locking attribute table
        del record
        del outCursor

        #The discrete intervals of the most current values are converted into an array
        currentArray=np.zeros(rowCount)
        for rowInd,row in enumerate(arcpy.SearchCursor(output)):
            currentArray[rowInd]=row.getValue(intfield)-1

        #A matrix with columns representing each discrete interval is produced.
        #The transition matrix is used to give each element in this matrix the probability of the most recent value
        #in the same row entering the discrete interval of the column in the next iteration of the time series
        tranmatrix = generate_transition_matrix(transitionArray,discrete)
        condprobArray=np.zeros([rowCount,discrete])
        for r in range(rowCount):
            for i in range(discrete):
                condprobArray[r][i] = tranmatrix[currentArray[r]][i]

        #The highest probabilities of each discrete interval column are compiled as indices in a new array
        maxprobArray = np.argmax(condprobArray,axis=1)

        #The final field created is an array of predicted values for the next Markovian iteration of the time series.
        predictfield = "Predicted"
        arcpy.AddField_management(output,predictfield,"DOUBLE",20,5)
        outCursor=arcpy.UpdateCursor(output)
        for index, record in enumerate(outCursor):
            predicted_value=(maxprobArray[index]+1)*global_interval
            record.setValue(predictfield,predicted_value)
            outCursor.updateRow(record)

        del record
        del outCursor

        dts = []
        for i in range(len(tranmatrix[0])):
            dts.append(tuple(["Interval"+str(i),"f8"]))

        struct_array=np.core.records.fromarrays(tranmatrix.transpose(),np.dtype(dts))
        arcpy.da.NumPyArrayToTable(struct_array,outputTable)

except Exception as e:
        # If unsuccessful, end gracefully by indicating why
        arcpy.AddError('\n'+"Script failed because: \t\t"+e.message)
        # ... and where
        exceptionreport=sys.exc_info()[2]
        fullermessage=traceback.format_tb(exceptionreport)[0]
        arcpy.AddError("at this location: \n\n"+fullermessage+"\n")

