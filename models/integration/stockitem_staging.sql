{{ config(materialized='view') }}
select *
from {{ source('integration', 'StockItem_Staging') }}
