variable "ibmcloud_api_key" {
  description = "Your IBM Cloud API key (set this via env or tfvars)"
  type        = string
  sensitive   = true
}

variable "region" {
  description = "IBM Cloud region for the cluster"
  type        = string
  default     = "eu-de"
}

variable "cluster_name" {
  description = "Name of your existing OpenShift cluster"
  type        = string
}

variable "pool_name" {
  description = "Name for the new GPU worker pool"
  type        = string
  default     = "gpu-pool"
}

variable "flavor" {
  description = "VSI flavor for GPU nodes (e.g. g2.8x64)"
  type        = string
  default     = "g2.8x64"
}

variable "worker_count" {
  description = "Number of GPU nodes to create"
  type        = number
  default     = 1
}

variable "zone" {
  description = "Failure domain / zone for the pool"
  type        = string
  default     = "eu-de-1"
}

variable "node_labels" {
  description = "Labels to apply to GPU nodes"
  type        = map(string)
  default     = {
    "role" = "gpu"
  }
}
