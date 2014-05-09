Please find a data description below.

You will notice considerable drops in bank holiday weeks. I would either remove them for now or replace them with something sensible. We typically use some historical knowledge about bank holidays to rescale the counts for such weeks, but I would not spend much time on that at this stage.

If you have got any questions or require another view onto the data set, please let me know. I have the raw data (a row for every single job).  

Martin

======================================
Description:
The data represents job counts aggregated by week and other attribute values. You can use the data to construct a timeseries of job counts for every attribute-value combination.

However, please note that there are no jobs for certain attribute-value combinations in some of the weeks. Rather than having a job_count of zero, these attribute-value combinations are missing in the data.

The attributes REGION and SUBREGION are hierarchical in the sense that the job count of a REGION should be the sum of job counts in the SUBREGIONS of REGION. For instance, if REGION was London then associated SUBREGIONS would be different parts of London. 

Furthermore there are mappings between other attributes, which are more or less uncertain.

Sometimes, attribute values are not known which should be denoted with the string NULL, unless I have missed to replace some missing values.

-------------------------------
Attributes in columns:

- WEEK_START_DATE: date of the first day in the week (should be Saturday)
- WEEK_INDEX: index of the week starting with 1
- PRODUCT_DESC_1: first product description related to the job
- PRODUCT_DESC_2: second product description related to the job. Note, there is a relationship between the values in PRODUCT_DESC_1 and PRODUCT_DESC_2.
- REGION: geographical region of the job
- SUBREGION: geographical sub-region of REGION
- JOB_TYPE: type of job
- CUSTOMER: name of customer
- PRIOR_NETWORK_LOCATION: assumed job location before job started
- POST_NETWORK_LOCATION: confirmed job location after job finished (different classification than PRIOR_NETWORK_LOCATION)
- JOB_COUNT: number of jobs with all the attribute values above

Dr Martin Spott
Chief Scientist, Analytics
BT Research and Technology