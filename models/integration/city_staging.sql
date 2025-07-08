{{ config(materialized='view') }}
select *
from {{ source('integration', 'City_Staging') }}
