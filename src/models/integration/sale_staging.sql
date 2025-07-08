{{ config(materialized='view') }}
select * from {{ source('integration', 'Sale_Staging') }}
