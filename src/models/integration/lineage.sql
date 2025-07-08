{{ config(materialized='view') }}
select * from {{ source('integration', 'Lineage') }}
