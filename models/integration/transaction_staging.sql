{{ config(materialized='view') }}
select *
from {{ source('integration', 'Transaction_Staging') }}
