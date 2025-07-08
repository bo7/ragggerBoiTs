{{ config(materialized='view') }}
select * from {{ source('integration', 'Order_Staging') }}
