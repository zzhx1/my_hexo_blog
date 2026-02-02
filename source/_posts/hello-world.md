---
title: Clash-CLI 部署与跨机器代理使用全指南
date: 2025-09-08
updated: 2026-01-15
tags:
  - Clash
  - 代理配置
  - Linux
  - 命令行工具
  - 网络工具
category: 技术教程
---
本文详细讲解 `clash-cli` 工具的安装、云服务器代理配置，以及本地机器如何一键复用云服务器的代理服务，全程附实操命令和避坑要点。

## 一、Clash-CLI 安装（云服务器端）
`clash-cli` 是 Clash 代理的命令行管理工具，支持快速安装、更新订阅、启停代理，推荐在 Linux 云服务器（如阿里云/腾讯云）部署。

### 方式一：Python 包安装（推荐，易维护）
```bash
# 1. 安装 clash-cli 核心工具（确保服务器已安装 Python/pip）
pip install clash-cli

# 2. 初始化（解决 sudo 权限问题，非必需但建议执行）
clash-cli init

# 3. 安装 Clash 系统服务（需 sudo 权限，生成 systemd 服务）
sudo clash-cli install

# 4. 启动代理（验证安装）
clash-cli on
```

### 方式二：Shell 脚本安装（传统方式，兼容低版本系统）
```bash
git clone --branch main --depth 1 https://github.com/whillhill/clash-cli.git \
  && cd clash-cli \
  && sudo bash install.sh
```

## 二、Clash-CLI 基础使用（云服务器端）
### 1. 订阅管理（核心：更新代理节点）
```bash
# 首次更新：使用自定义订阅链接
clash-cli update https://your-subscription-url.com

# 后续更新：复用上次的订阅链接（无需重复输入）
clash-cli update

# 查看更新日志（确认订阅是否生效）
clash-cli update log
```

### 2. 代理服务启停与状态查看
```bash
# 启动代理服务
clash-cli on

# 停止代理服务
clash-cli off

# 查看服务状态（关键：确认 mihomo.service 是否运行）
clash-cli status
```
> 正常状态示例：
> ```
> ● mihomo.service - mihomo Daemon, A[nother] Clash Kernel.
>    Loaded: loaded (/etc/systemd/system/mihomo.service; enabled; vendor preset: enabled)
>    Active: active (running) since Mon 2025-01-27 10:30:15 CST; 2h 15min ago
> ```

## 三、云服务器代理配置（允许本地访问）
默认情况下 Clash 仅监听本机（127.0.0.1），需修改配置开启局域网访问，并添加认证防止滥用。

### 1. 确认端口监听状态
```bash
# 查看 7890 端口监听范围（默认是 127.0.0.1，仅本机可访问）
sudo netstat -tulpn | grep 7890
```

### 2. 修改 Clash 配置文件
#### 步骤1：找到配置文件路径
通过 `clash-cli status` 输出找到配置文件（示例：`/opt/clash/runtime.yaml`）。

#### 步骤2：编辑配置文件
```bash
sudo vim /opt/clash/runtime.yaml
```
修改以下核心配置（添加局域网访问+用户认证）：
```yaml
# 允许局域网访问（关键：开启后本地机器才能连接）
allow-lan: true

# 用户认证（必填！防止代理暴露公网被他人滥用）
authentication:
  - "user:123123" # 格式："用户名:密码"，可自定义

# 端口配置（默认 7890，无需修改）
port: 7890
socks-port: 7891
redir-port: 7892
```

#### 步骤3：重启服务使配置生效
```bash
sudo systemctl restart mihomo
# 验证端口监听（显示 :::7890 表示监听所有IP）
sudo netstat -tulpn | grep 7890
```
> 正常输出示例：
> ```
> tcp6       0      0 :::7890                 :::*                    LISTEN      57850/mihomo
> udp6       0      0 :::7890                 :::*                                57850/mihomo
> ```

### 3. 开放云服务器端口（必做！）
登录云服务器控制台（如阿里云/腾讯云），在「安全组」中放行 7890 端口（TCP/UDP 协议），否则本地无法连接。

## 四、本地机器使用云服务器代理
### 1. 验证代理连通性
先通过 `curl` 测试是否能正常访问外网（替换为你的服务器IP/账号密码）：
```bash
# 测试访问 Google，验证代理是否生效
curl -x http://user:123123@xxx.xxx.xxx.xx:7890 https://google.com -v
```
> 成功标志：返回 `301 Moved` 或 Google 页面内容，无连接超时/拒绝错误。

### 2. 一键代理脚本（本地快捷使用）
编写脚本 `proxyctl`，实现「一键开启/关闭/查看代理」，无需手动输配置。

#### 步骤1：创建脚本文件
```bash
vim /home/zzhxx/workspace/tools/proxyctl
```
粘贴以下内容（已适配你的配置，可直接用）：
```bash
#!/bin/bash
set -e # 遇到错误立即退出

# ========== 自定义配置（替换为自己的信息）==========
PROXY_USER="user"
PROXY_PASS="123123"
PROXY_IP="xxx.xxx.xxx.xx"
PROXY_PORT="7890"
# ================================================

PROXY_URL="http://${PROXY_USER}:${PROXY_PASS}@${PROXY_IP}:${PROXY_PORT}"

# 执行逻辑
case "$1" in
    on)
        # 开启代理：清空旧变量 + 设置新变量
        unset HTTPS_PROXY HTTP_PROXY ALL_PROXY
        export HTTPS_PROXY="${PROXY_URL}"
        export HTTP_PROXY="${PROXY_URL}"
        export ALL_PROXY="${PROXY_URL}"
        echo -e "\033[32m✅ 代理已开启\033[0m"
        echo "当前代理配置：${PROXY_URL}"
        ;;
    off)
        # 关闭代理：清空所有代理变量
        unset HTTPS_PROXY HTTP_PROXY ALL_PROXY
        echo -e "\033[31m❌ 代理已关闭\033[0m"
        ;;
    status)
        # 查看当前代理状态
        echo -e "📌 当前代理状态："
        echo "HTTPS_PROXY: ${HTTPS_PROXY:-未设置}"
        echo "HTTP_PROXY: ${HTTP_PROXY:-未设置}"
        echo "ALL_PROXY: ${ALL_PROXY:-未设置}"
        ;;
    test)
        # 新增：测试代理连通性
        echo -e "🔍 测试代理连通性..."
        if curl -s -x "${PROXY_URL}" https://google.com > /dev/null; then
            echo -e "\033[32m✅ 代理可用\033[0m"
        else
            echo -e "\033[31m❌ 代理不可用\033[0m"
            exit 1
        fi
        ;;
    *)
        # 帮助信息
        echo -e "📚 用法："
        echo "  proxyctl on      - 开启代理"
        echo "  proxyctl off     - 关闭代理"
        echo "  proxyctl status  - 查看代理状态"
        echo "  proxyctl test    - 测试代理连通性"
        ;;
esac
```

#### 步骤2：添加执行权限
```bash
chmod +x /home/zzhxx/workspace/tools/proxyctl
```

#### 步骤3：设置别名（永久生效）
```bash
# 临时生效（当前终端）
alias proxyctl="sh /home/zzhxx/workspace/tools/proxyctl"

# 永久生效（所有终端），添加到 ~/.bashrc 或 ~/.zshrc
echo "alias proxyctl='/home/zzhxx/workspace/tools/proxyctl'" >> ~/.bashrc
source ~/.bashrc
```

#### 步骤4：使用示例
```bash
# 开启代理
proxyctl on

# 查看状态
proxyctl status

# 测试连通性
proxyctl test

# 关闭代理
proxyctl off
```

### 3. 额外实用脚本（可选）
停止并删除 Docker 容器的快捷脚本（补充）：
```bash
#!/bin/bash
# 检查参数是否传入
if [ -z "$1" ]; then
    echo -e "\033[31m❌ 请传入容器名/ID\033[0m"
    echo "用法：docker-rm 容器名/ID"
    exit 1
fi

name=$1
echo -e "🔧 停止并删除容器：${name}"
docker stop ${name} && docker rm ${name}
echo -e "\033[32m✅ 操作完成\033[0m"
```

## 五、避坑要点
1. **权限问题**：修改 Clash 配置文件需 `sudo`，否则保存失败；
2. **安全风险**：开启 `allow-lan: true` 后必须加 `authentication`，否则代理可能被全网滥用；
3. **端口放行**：云服务器安全组未开 7890 端口，本地会提示「连接拒绝」；
4. **脚本生效**：设置别名后需 `source ~/.bashrc`，否则新终端不生效；
5. **代理范围**：脚本仅对当前终端生效，如需全局代理需配置系统网络。

---

### 总结
1. 核心流程：云服务器安装 `clash-cli` → 配置局域网访问+认证 → 开放端口 → 本地用脚本一键复用；
2. 关键避坑：开启局域网访问必须加用户认证，云服务器安全组需放行 7890 端口；
3. 效率优化：本地脚本实现「开启/关闭/测试」一体化，无需重复输入复杂代理地址。
