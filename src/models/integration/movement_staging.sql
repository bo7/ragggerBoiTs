{{ config(materialized='view') }}
select * from {{ source('integration', 'Movement_Staging') }}
