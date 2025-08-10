# AWS Cloud Setup for RAG System Deployment

## Prerequisites

### Download AWS CLI
Follow the official installation guide:
```bash
https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
```

---

## 1. IAM User Setup

### Create IAM User
1. Search for **IAM** in AWS Console
2. Click on **Users**
3. Click **Create user**
4. Enter the **user name** (e.g., `rag-deployment-user`)

### Attach Policies
Attach the following policies to the user:
- `AdministratorAccess`
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess`

5. Click **Next** and then **Create user**

### Generate Access Keys
1. Go to the created user
2. Click on **Security credentials** tab
3. Click **Create access key**
4. Select Use Case: **Local code**
5. Click **Next** and **Create access key**

### Configure AWS CLI
Run the following command and enter your credentials:
```bash
aws configure
```
```
AWS Access Key ID [****************ZPUP]: your-access-key-id
AWS Secret Access Key [****************P184]: your-secret-access-key
Default region name [us-east-1]: us-east-1
Default output format [None]: json
```

---

## 2. S3 Bucket Setup

1. Search for **S3** in AWS Console
2. Click **Create bucket**
3. Enter a **globally unique bucket name** (e.g., `your-company-rag-artifacts-2024`)
4. Keep all other settings as default
5. Click **Create bucket**

---

## 3. ECR Repository Setup

1. Search for **ECR** in AWS Console
2. Click **Create repository**
3. Enter **Repository name** (e.g., `abu-dhabi-rag-system`)
4. Keep settings as default
5. Click **Create**

**Note:** Copy the repository URI for later use in GitHub secrets.

---

## 4. EC2 Instance Setup

### Launch Instance
1. Search for **EC2** in AWS Console
2. Click **Launch instance**
3. Configure the following:
   - **Name**: `rag-system-server`
   - **OS Image**: Ubuntu Server 22.04 LTS
   - **Instance Type**: t3.medium (or as per requirement)
   - **Key pair**: Create new or select existing
   - **Security Group**: Configure as below

### Security Group Configuration
Select the following options:
- ✅ **Allow SSH traffic from** → Anywhere (0.0.0.0/0)
- ✅ **Allow HTTPS traffic from the internet**
- ✅ **Allow HTTP traffic from the internet**

### Storage Configuration
- Configure storage as per requirement (minimum 20GB recommended)

4. Click **Launch instance**

### Configure Additional Ports
After instance creation:
1. Select your **Instance ID**
2. Go to **Security** tab
3. Click on the **Security group** (e.g., sg-0967b9f6d0ee3be7c)
4. Click **Edit inbound rules**
5. Click **Add rule** and configure:
   - **Type**: Custom TCP
   - **Port range**: 8501 (Streamlit)
   - **Source**: 0.0.0.0/0
6. Add another rule for Phoenix:
   - **Type**: Custom TCP
   - **Port range**: 6006 (Phoenix)
   - **Source**: 0.0.0.0/0
7. Click **Save rules**

---

## 5. GitHub Secrets Configuration

Navigate to your repository **Settings** → **Secrets and variables** → **Actions**

Add the following secrets:

### AWS Credentials
- `AWS_ACCESS_KEY_ID`: Your IAM user access key
- `AWS_SECRET_ACCESS_KEY`: Your IAM user secret key
- `AWS_DEFAULT_REGION`: us-east-1 (or your preferred region)

### AWS Resources
- `AWS_ECR_LOGIN_URI`: Your ECR repository URI (e.g., 123456789012.dkr.ecr.us-east-1.amazonaws.com)
- `ECR_REPOSITORY_NAME`: Your ECR repository name (e.g., abu-dhabi-rag-system)

### API Keys
- `GROQ_API_KEY`: Your Groq API key
- `ASTRA_DB_API_ENDPOINT`: Your Astra DB endpoint
- `ASTRA_DB_APPLICATION_TOKEN`: Your Astra DB token
- `ASTRA_DB_KEYSPACE`: Your Astra DB keyspace name
- `HF_TOKEN`: Your Hugging Face token
- `COHERE_API_KEY`: Your Cohere API key

---

## 6. Install Docker on EC2

SSH into your EC2 instance and run the following commands:

### Update System
```bash
sudo apt-get update -y
sudo apt-get upgrade -y
```

### Install Docker
```bash
# Download Docker installation script
curl -fsSL https://get.docker.com -o get-docker.sh

# Install Docker
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker ubuntu

# Apply group changes
newgrp docker

# Verify installation
docker --version
```

---

## 7. GitHub Actions Self-Hosted Runner Setup

### Configure Runner in GitHub
1. Go to your repository **Settings** → **Actions** → **Runners**
2. Click **New self-hosted runner**
3. Select **Linux** as Runner image
4. Follow the commands provided (updated versions below)

### Setup Commands for EC2

#### Create Runner Directory
```bash
mkdir actions-runner && cd actions-runner
```

#### Download Latest Runner Package
```bash
# Download (check GitHub for latest version)
curl -o actions-runner-linux-x64-2.319.1.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.319.1/actions-runner-linux-x64-2.319.1.tar.gz
```

#### Validate Hash (Optional)
```bash
echo "3f6efb7488a183e291fc2c62876e14c9ee732864173734facc85a1bfb1744464  actions-runner-linux-x64-2.319.1.tar.gz" | shasum -a 256 -c
```

#### Extract Installer
```bash
tar xzf ./actions-runner-linux-x64-2.319.1.tar.gz
```

#### Configure Runner
```bash
# Replace with your repository URL and token from GitHub
./config.sh --url https://github.com/YOUR-USERNAME/YOUR-REPOSITORY --token YOUR-REGISTRATION-TOKEN
```

#### Start Runner
```bash
# Run once
./run.sh

# Or run as service (recommended for production)
sudo ./svc.sh install
sudo ./svc.sh start
```

---

## 8. Deployment Verification

### Check Services
After deployment, verify the following:
- **Streamlit App**: `http://your-ec2-public-ip:8501`
- **Phoenix Dashboard**: `http://your-ec2-public-ip:6006`
- **EC2 Instance**: Running and accessible
- **GitHub Actions**: Runner connected and workflows executing

### Troubleshooting
- Check EC2 security groups for proper port configuration
- Verify environment variables in GitHub secrets
- Monitor GitHub Actions logs for deployment issues
- Check EC2 instance logs: `sudo docker logs container-name`

---

## Security Notes

⚠️ **Important Security Considerations:**
- Never commit API keys to your repository
- Use GitHub secrets for all sensitive information
- Regularly rotate your AWS access keys
- Consider using AWS IAM roles instead of access keys for production
- Monitor your AWS usage and costs
- Enable AWS CloudTrail for audit logging

---

## Cost Optimization

💰 **Cost Management Tips:**
- Use t3.micro for development (eligible for free tier)
- Stop EC2 instances when not in use
- Monitor S3 storage costs
- Set up AWS billing alerts
- Use ECR lifecycle policies to manage image storage