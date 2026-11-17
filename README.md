# AstrBot 鸟类识别插件

对接本地自建的鸟类识别服务（懂鸟 v2 协议），在聊天里直接认鸟。发一张照片或一段录音，
插件回你中文名对照英文名、裁剪后的鸟图或声谱图，以及本次处理耗时。

- 命令：`/birds`［地点］—— 识别图片或录音里的鸟
- 管理员命令：`/birdstatus` —— 查看后端与归档状态
- 依赖：AstrBot v4.0.0 及以上、一个可访问的懂鸟协议后端；音频处理需要 `ffmpeg`

## 特性

- **图片与录音都支持**：直接发附件，或引用一条带图片/录音的消息再发 `/birds`。
- **中英对照**：后端对鸟类只给学名，插件用内置名称表折算成「中文名 / English / 学名」；
  名称表合并了两个公开名录，覆盖 11587 条，中文命中率 99% 以上。
- **保留门禁输出**：回复里保留检测框坐标与门禁置信度，裁剪图按检测框生成。
- **图片给裁剪图，音频给声谱图**：裁剪图最多 3 张，声谱图 900×420，纯 numpy 绘制；
  回复图编码成 JPEG，一条消息里的几张图合计只有几百 KB。
- **输出处理时长**：每次回复末尾都有 `耗时 xx ms`。
- **不泄露内部信息**：回复和日志里不会出现模型名、设备名、文件路径或后端地址。
- **异步并发 + 忙时拒绝**：插件侧有在途配额，后端队列满时立刻回「现在有点忙」，不排队堆积。
- **归档轮转**：识别用过的图片和音频压缩进 `tar.gz` 归档，按天数和总体积轮转，可用于后续训练。
- **单发图片不进大模型**：回复直接发出并终止事件，语言模型不会收到这条消息。

## 安装

1. 把整个目录放进 AstrBot 的插件目录：

   ```bash
   git clone https://github.com/yblpoi/astrbot_plugin_birdid
   mv astrbot_plugin_birdid /path/to/AstrBot/data/plugins/
   ```

2. 安装依赖（AstrBot 用的解释器或容器）：

   ```bash
   pip install -r requirements.txt
   ```

3. 在 AstrBot 管理面板启用插件，然后在插件配置里填后端地址。

4. 重启 AstrBot 或重载插件。启动日志出现 `Bird identification plugin ready with N names.`
   就说明名称表加载成功。

### 后端要求

插件的默认后端是 `http://127.0.0.1:8080`，即本机跑的懂鸟协议服务。该服务需要：

- `POST /dongniao` 接受 `image` / `sound` multipart 上传，返回 `[code, payload]`；
- `GET /healthz` 返回 `{"status": ..., "version": ..., "pending": ..., "max_queue": ...}`；
- 异步受理：先回 `[1001, resultid]`，再轮询 `resultid` 拿最终结果。

如果后端跑在另一台机器或 Docker 容器里，把 `backend_url` 改成插件**能访问到**的地址。
AstrBot 以 Docker 部署、后端跑在宿主机时，容器里写 `127.0.0.1` 指向容器自身，需要改成宿主机在
容器网桥上的地址（一般是 `172.18.0.1`，可用 `docker network inspect <网络名>` 确认）。

懂鸟官方 API（`https://ai.open.hhodata.com/api/v2`）与本协议不完全相同：官方只做同步返回、
没有 `/healthz`，`/birdstatus` 会显示为离线。本插件按自建服务的异步协议设计，
如需对接官方接口需要另外适配。

## 使用方法

### `/birds`

```
/birds                       # 识别本条消息里附带的图片或录音
/birds 云南                   # 附带地点先验
/birds 广东省                 # 省 / 市 / 国家都可以
/birds China                  # 英文国名也可以
```

也可以引用一条消息：

```
（先引用那张图或那段语音）
/birds 四川
```

地点可以带空格，`/birds 中国 云南` 会把整段当作一个地点解析。它的作用是让后端叠加
地理季节先验，提高同域近似种的分辨率。插件只映射后端**确实有坐标**的地区；后端没有
坐标的省份会被忽略而不是硬掰到省会，此时回复里不会出现「地点先验」这一行。

### `/birdstatus`（仅管理员）

```
鸟类识别后端
状态：在线
版本：0.1.0
后端队列：0/32
后端负载：ok
本插件：排队 0，处理中 0
归档：待打包 1.2 MB，已归档 48.7 MB，批次 6
名称库：11587 条
```

非管理员触发会收到「该指令仅限管理员使用。」

## 输出示例

```
识别结果
1. 麻雀 / Eurasian Tree Sparrow / Passer montanus  96.4%
   检测框 412,208-903,741  门禁 97.1%
2. 山麻雀 / Russet Sparrow / Passer cinnamomeus  2.1%
   检测框 412,208-903,741  门禁 97.1%

耗时 218 ms
```

录音的回复结构相同，只是没有检测框那一行，并附一张声谱图。

## 配置

在 AstrBot 管理面板的插件配置里填写，全部有默认值：

| 键 | 默认值 | 说明 |
| --- | --- | --- |
| `backend_url` | `http://127.0.0.1:8080` | 后端地址 |
| `api_key` | 空 | 后端要求鉴权时填；留空则不发送 |
| `api_key_header` | `api_key` | 携带 key 的请求头名 |
| `max_in_flight` | `2` | 本插件同时处理的请求数，超出的立刻回忙 |
| `backend_concurrency` | `4` | 与后端之间的并发上限 |
| `request_timeout_s` | `30` | 单次识别的超时时间（秒） |
| `ffmpeg_path` | 空 | 留空则用 `PATH` 里的 `ffmpeg` |
| `retention_days` | `7` | 归档保留天数 |
| `archive_max_mb` | `2048` | 归档总体积上限（MB） |
| `archive_interval_s` | `900` | 归档整理间隔（秒） |

`max_in_flight` 与 `backend_concurrency` 是两道闸：前者限制聊天侧积压，后者限制插件对后端的
连接数。后端本身还有自己的队列，队列打满时会回 `server busy`，插件据此立刻返回繁忙提示。

## 数据与隐私

- **不长期保留对话**：每条请求的临时文件写在系统临时目录下的独立子目录，回复发完就删除。
- **归档用于训练**：识别成功的图片/录音、置信度、检测框、地点名和处理耗时会写进归档目录
  （`data/plugin_data/astrbot_plugin_birdid/bird_archive/`），按 `retention_days` 与
  `archive_max_mb` 自动轮转，过期批次直接删除。
- **不落敏感字段**：归档写入前会剥掉绝对路径和控制字符，`api_key`、`token`、`password` 一类
  字段整个丢弃。
- **回复不含内部信息**：不出现模型名、设备名、路径、后端地址或任何堆栈。

## 名称表

`bird/data/bird_names.tsv.gz` 是「归一化学名 → 中文名 → 英文名」的 gzip TSV，11587 行，
约 200 KB。来源与许可见 [NOTICE.md](NOTICE.md)。学名在查表前会归一化（小写、下划线转空格、
压缩空白），所以 `Passer_montanus`、`passer  montanus` 都能命中同一行。

图片分类器报告的物种名是英文俗名而不是学名（例如 `Cerulean Kingfisher` 会被写成
`Small Blue Kingfisher`），所以插件加载时还会建反向索引：先按归一化学名查，查不到再按归一化
英文名查。归一化会抹平大小写、下划线、连字符和撇号差异（`Blyths Kingfisher` 与
`Blyth's Kingfisher` 视为同一行）。

`bird/data/bird_aliases.tsv.gz` 是补充别名表（964 行，约 19 KB）：把分类器里那些与名录拼写
不一致的类名，按「分类器类序号 → 名录学名」映射到主表的行。别名表只用已有键的空位，
永远不会覆盖主表已有的条目。

名称表缺失或损坏不会让插件失效：查不到的学名会回退显示后端返回的原文。

## 开发

```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio ruff

ruff check .
ruff format --check .
python -m pytest tests -q
```

代码结构：

```
main.py              AstrBot 入口，两个命令与事件收尾
bird/client.py       懂鸟协议客户端（提交 + 轮询 + 错误码）
bird/media.py        从消息里取媒体、图片规范化、音频转码
bird/service.py      一次识别的编排：并发闸、裁剪图、声谱图、归档
bird/render.py       裁剪与声谱图绘制（只依赖 Pillow 与 numpy）
bird/names.py        学名到中英文名的查表
bird/regions.py      地点文本到后端地区码
bird/format.py       回复文本渲染
bird/archive.py      归档暂存、打包、轮转
bird/data/           名称表与别名表
tests/               单元测试
```

## 已知限制

- 后端只解 16 bit PCM WAV，其它格式一律先用 `ffmpeg` 转成 32 kHz 单声道，没装 `ffmpeg`
  时录音识别不可用（图片识别不受影响）。
- 一条消息只识别第一个图片或第一个录音，多附件时按图片优先。
- 地点先验只能映射后端 `regions` 表里有的地区码，其它省份不带先验。
- 分类器有 11 个类名对不上任何名录条目（多为杂交种或分类学新拆分），这些条目的中文名
  留空、只显示后端返回的英文名。

## 许可

插件代码为 MIT，第三方数据许可见 [NOTICE.md](NOTICE.md)。