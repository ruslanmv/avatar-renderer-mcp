output "gpu_pool_id" {
  description = "The ID of the created GPU worker pool"
  value       = ibm_container_vpc_cluster_worker_pool.gpu_pool.id
}

output "gpu_pool_name" {
  description = "The name of the GPU worker pool"
  value       = ibm_container_vpc_cluster_worker_pool.gpu_pool.name
}
