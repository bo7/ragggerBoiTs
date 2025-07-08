{{ config(materialized='view') }}
select * from {{ source('integration', 'ETL Cutoff') }}
