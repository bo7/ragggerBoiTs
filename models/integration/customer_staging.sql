{{ config(materialized='view') }}
select *
from {{ source('integration', 'Customer_Staging') }}
