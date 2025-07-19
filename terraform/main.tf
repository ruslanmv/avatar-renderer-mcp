terraform {
  required_providers {
    ibm = {
      source  = "IBM-Cloud/ibm"
      version = "1.20.0"
    }
  }
}

provider "ibm" {
  ibmcloud_api_key = var.ibmcloud_api_key
  region           = var.region
}

# Data: fetch existing OpenShift cluster
data "ibm_container_vpc_cluster" "cluster" {
  cluster_name = var.cluster_name
}

# Create a new GPU worker pool on that cluster
resource "ibm_container_vpc_cluster_worker_pool" "gpu_pool" {
  cluster     = data.ibm_container_vpc_cluster.cluster.id
  name        = var.pool_name
  flavor      = var.flavor
  count       = var.worker_count
  worker_zone = var.zone

  labels = var.node_labels

  # Optional taint
  taints = [{
    key    = "dedicated"
    value  = "gpu"
    effect = "NoSchedule"
  }]
}
