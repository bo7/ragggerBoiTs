{{ config(materialized='view') }}
select * from {{ source('integration', 'StockHolding_Staging') }}
