{{ config(materialized='view') }}
select *
from {{ source('integration', 'Purchase_Staging') }}
