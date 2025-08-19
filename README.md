# 🩺 Vaccine Hesitancy Analysis Platform

A comprehensive AI-powered platform for analyzing vaccine hesitancy data, providing insights for health affairs professionals through multiple specialized agents and an intuitive web interface.

## 🌟 Features

### 📊 **Health Board Analysis**
- Geographic visualization of vaccine hesitancy by UK regions
- Interactive maps with Mapbox integration
- Demographic breakdowns (age, sex, ethnicity, religion)
- Trend analysis over time

### 📱 **Social Media Analysis**
- X (Twitter) sentiment analysis for vaccine-related discussions
- Timeline trends and engagement metrics
- Influencer analysis and viral content detection
- Real-time social media monitoring

### 📚 **Vaccine Resources**
- Vector-based document search using Qdrant
- AI-powered query understanding
- Medical terminology expansion
- Comprehensive vaccine guideline access

### 🎤 **Voice Analysis**
- Audio file processing and transcription
- Vaccine hesitancy detection in speech
- Keyword extraction and sentiment analysis
- Support for multiple audio formats

## 🏗️ Architecture

The platform consists of multiple specialized AI agents working together:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   VH_Insights   │    │   VH_Resource   │
│   (Next.js)     │◄──►│   Agent         │◄──►│   Agent         │
│   Port: 8010    │    │   Port: 8003/5  │    │   Port: 8002/6  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │   VH_Voice      │              │
         │              │   Analyzer      │              │
         │              │   Port: 8004    │              │
         │              └─────────────────┘              │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   VH_X_Analysis │
                    │   Agent         │
                    │   Port: 8001    │
                    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.8+ (for local development)
- Node.js 18+ (for frontend development)

### Option 1: Docker Compose (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Vaccine_Hesitancy
   ```

2. **Set up environment variables**
   ```bash
   # Copy example files for each agent
   cp agents/VH_Insights_Agent/env.example agents/VH_Insights_Agent/.env
   cp agents/VH_Resource_Agent/env.example agents/VH_Resource_Agent/.env
   cp agents/VH_Voice_Analyzer_Agent/env.example agents/VH_Voice_Analyzer_Agent/.env
   cp agents/VH_X_Analysis_Agent/env.example agents/VH_X_Analysis_Agent/.env
   cp frontend/env.example frontend/.env.local
   
   # Edit each .env file with your API keys
   nano agents/VH_Insights_Agent/.env
   nano agents/VH_Resource_Agent/.env
   nano agents/VH_Voice_Analyzer_Agent/.env
   nano agents/VH_X_Analysis_Agent/.env
   nano frontend/.env.local
   ```

3. **Start all services**
   ```bash
   docker-compose up --build
   ```

4. **Access the platform**
   - Frontend: http://localhost:8010
   - VH Insights REST: http://localhost:8005
   - VH Resource REST: http://localhost:8006
   - VH Voice Analyzer: http://localhost:8004
   - VH X Analysis: http://localhost:8001

### Option 2: Individual Agent Setup

#### Frontend Setup
```bash
cd frontend
npm install
cp env.example .env.local
# Edit .env.local with your Mapbox token
npm run dev
```

#### VH Insights Agent
```bash
cd agents/VH_Insights_Agent
pip install -r requirements.txt
cp env.example .env
# Edit .env with your credentials
python vh_agent.py          # Chat agent
python vh_rest_agent.py     # REST API agent
```

#### VH Resource Agent
```bash
cd agents/VH_Resource_Agent
pip install -r requirements.txt
cp env.example .env
# Edit .env with your credentials
python vaccine_resource_agent.py          # Chat agent
python vaccine_resource_rest_agent.py    # REST API agent
```

#### VH Voice Analyzer Agent
```bash
cd agents/VH_Voice_Analyzer_Agent
pip install -r requirements.txt
cp env.example .env
# Edit .env with your credentials
python vh_voice_analyzer_rest_agent.py
```

#### VH X Analysis Agent
```bash
cd agents/VH_X_Analysis_Agent
pip install -r requirements.txt
cp env.example .env
# Edit .env with your credentials
python x_analysis_rest_agent.py
```

## 📊 Data Loaders

The platform includes data loaders for processing vaccine hesitancy data from Excel files into the Supabase database.

### Setup Data Loaders

1. **Install dependencies**
   ```bash
   pip install pandas sqlalchemy psycopg2-binary openpyxl
   ```

2. **Configure database connection**
   - Ensure your `.env` file has the correct Supabase database credentials
   - The data loaders use SQLAlchemy to connect to PostgreSQL

3. **Run data loaders**
   ```bash
   cd utils/data_loaders
   
   # Load specific tables
   python t1a_data_loader.py  # Age group data
   python t2_data_loader.py   # Region data
   python t3_data_loader.py   # Sex data
   # ... and so on for all 18 data loaders
   
   # Or run all at once (if you have a script)
   for file in t*.py; do python "$file"; done
   ```

### Data Loader Structure

Each data loader processes a specific Excel sheet and loads it into corresponding database tables:
- **t1a, t1b**: Age group analysis
- **t2**: Regional analysis
- **t3**: Sex-based analysis
- **t4-t18**: Various demographic and trend analyses

## 🔧 Configuration

### Required Environment Variables

#### VH Insights Agent
```bash
SUPABASE_ACCESS_TOKEN=sbp_your-token
SUPABASE_PROJECT_ID=your-project-id
SUPABASE_DB_PASSWORD=your-db-password
ASI1_API_KEY=sk_your-asi1-key
```

#### VH Resource Agent
```bash
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-key
OPENAI_API_KEY=sk-proj-your-openai-key
ASI1_API_KEY=sk_your-asi1-key
```

#### VH Voice Analyzer Agent
```bash
ASI1_API_KEY=sk_your-asi1-key
```

#### VH X Analysis Agent
```bash
X_API_KEY=your-x-api-key
X_API_SECRET=your-x-api-secret
X_ACCESS_TOKEN=your-x-access-token
X_ACCESS_SECRET=your-x-access-secret
X_BEARER_TOKEN=your-x-bearer-token
RAPIDAPI_KEY=your-rapidapi-key
SUPABASE_ACCESS_TOKEN=sbp_your-token
SUPABASE_PROJECT_ID=your-project-id
SUPABASE_DB_PASSWORD=your-db-password
ASI1_API_KEY=sk_your-asi1-key
```

#### Frontend
```bash
NEXT_PUBLIC_MAPBOX_TOKEN=pk_your-mapbox-token
```

### Getting API Keys

- **Supabase**: [https://supabase.com](https://supabase.com)
- **ASI1 (Fetch.ai)**: [https://innovationlab.fetch.ai/](https://innovationlab.fetch.ai/)
- **X (Twitter) API**: [https://developer.twitter.com/](https://developer.twitter.com/)
- **Qdrant**: [https://cloud.qdrant.io/](https://cloud.qdrant.io/)
- **OpenAI**: [https://platform.openai.com/](https://platform.openai.com/)
- **Mapbox**: [https://account.mapbox.com/](https://account.mapbox.com/)

## 🐳 Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `vh-insights-chat-agent` | 8003 | Chat-based vaccine hesitancy insights |
| `vh-insights-rest-agent` | 8005 | REST API for vaccine hesitancy insights |
| `vh-resource-chat-agent` | 8002 | Chat-based document search |
| `vh-resource-rest-agent` | 8006 | REST API for document search |
| `vh-voice-analyzer-agent` | 8004 | Voice analysis and transcription |
| `vh-x-analysis-agent` | 8001 | Social media analysis |
| `frontend` | 8010 | Next.js web application |

## 📱 API Endpoints

### VH Insights Agent (Port 8005)
- `GET /health` - Health check
- `POST /analyze` - Analyze vaccine hesitancy data
- `GET /schema` - Get database schema information

### VH Resource Agent (Port 8006)
- `GET /health` - Health check
- `POST /search` - Search vaccine documents
- `POST /upload` - Upload new documents

### VH Voice Analyzer Agent (Port 8004)
- `GET /health` - Health check
- `POST /analyze-audio` - Analyze audio files
- `POST /analyze-text` - Analyze transcribed text

### VH X Analysis Agent (Port 8001)
- `GET /health` - Health check
- `POST /analyze-trends` - Analyze social media trends
- `GET /sentiment` - Get sentiment analysis

## 🧪 Development

### Local Development Setup

1. **Clone and setup**
   ```bash
   git clone <repository-url>
   cd Vaccine_Hesitancy
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   ```

4. **Set up environment variables**
   ```bash
   # Copy and configure .env files for each component
   cp env.example .env
   # Edit with your credentials
   ```

5. **Run services individually**
   ```bash
   # Terminal 1: Frontend
   cd frontend && npm run dev
   
   # Terminal 2: VH Insights Agent
   cd agents/VH_Insights_Agent && python vh_rest_agent.py
   
   # Terminal 3: VH Resource Agent
   cd agents/VH_Resource_Agent && python vaccine_resource_rest_agent.py
   
   # Terminal 4: VH Voice Analyzer Agent
   cd agents/VH_Voice_Analyzer_Agent && python vh_voice_analyzer_rest_agent.py
   
   # Terminal 5: VH X Analysis Agent
   cd agents/VH_X_Analysis_Agent && python x_analysis_rest_agent.py
   ```

### Testing

```bash
# Test individual agents
curl http://localhost:8005/health  # VH Insights
curl http://localhost:8006/health  # VH Resource
curl http://localhost:8004/health  # VH Voice
curl http://localhost:8001/health  # VH X Analysis

# Test frontend
curl http://localhost:8010
```

## 🔍 Troubleshooting

### Common Issues

1. **Port conflicts**
   - Ensure ports 8001-8006 and 8010 are available
   - Check for other services using these ports

2. **Database connection issues**
   - Verify Supabase credentials in `.env` files
   - Check if Supabase project is active
   - Ensure database password is correctly URL-encoded

3. **API key errors**
   - Verify all required API keys are set
   - Check API key validity and rate limits
   - Ensure proper environment variable names

4. **Docker build issues**
   - Clear Docker cache: `docker system prune -a`
   - Rebuild without cache: `docker-compose build --no-cache`

### Health Checks

All services include health check endpoints:
```bash
curl http://localhost:8001/health  # VH X Analysis
curl http://localhost:8002/health  # VH Resource Chat
curl http://localhost:8003/health  # VH Insights Chat
curl http://localhost:8004/health  # VH Voice Analyzer
curl http://localhost:8005/health  # VH Insights REST
curl http://localhost:8006/health  # VH Resource REST
```

## 📈 Monitoring

### Logs
```bash
# View all service logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f vh-insights-agent
docker-compose logs -f frontend
```

### Performance
- Monitor API response times
- Check database connection pool usage
- Monitor memory and CPU usage in Docker containers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ for public health professionals**
