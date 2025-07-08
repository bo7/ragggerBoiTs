{{ config(materialized='view') }}
select * from {{ source('integration', 'TransactionType_Staging') }}
