"""
Audio Preprocessor - AWAITING DECISION
音频预处理器 - 等待决策

⚠️ 重要：AWS Bedrock 目前没有 ASR (语音转录) 模型

Bedrock 可用模型：
✓ Claude 3.5 Sonnet - LLM / VLM
✓ Titan Embeddings - 文本嵌入
✗ ASR (语音转录) - 不可用

需要您的决策：

选项 1: 使用 OpenAI Whisper API（推荐，最成熟）
   - 优点：准确率高，支持多种语言，有时间戳
   - 缺点：需要 OpenAI API Key
   - 实现：保持现有 OpenAI Whisper 实现

选项 2: 使用 AWS Transcribe
   - 优点：AWS 生态，与 Bedrock 同供应商
   - 缺点：调用方式不同，需要额外配置
   - 实现：需要新建 transcribe_client.py

选项 3: 使用其他供应商（Google Cloud Speech、Azure Speech 等）
   - 需要指定供应商和配置

选项 4: 本地 Whisper 模型（之前的实现）
   - 优点：离线可用
   - 缺点：需要 GPU，部署复杂
   - 不建议使用（违反 API-First 原则）

请告诉我您希望如何处理音频 ASR 功能？
推荐选项 1（OpenAI Whisper API），因为它是最成熟的方案。
"""

import os
import logging
from pathlib import Path
from typing import Any, Optional, List, Dict
from datetime import datetime
import io

from .base import BasePreprocessor, RawContent, parse_date

logger = logging.getLogger(__name__)

# ASR API 依赖
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class AudioProcessor(BasePreprocessor):
    """
    音频预处理器 - 等待 ASR 决策
    
    当前实现：使用 OpenAI Whisper API（临时方案，等待确认）
    
    TODO: 根据用户决策更新 ASR 方案
    """
    
    SUPPORTED_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.webm', '.mp4'}
    
    def __init__(
        self,
        asr_client=None,
        api_key: str = None,
        enable_diarization: bool = False,
        diarization_api_key: str = None
    ):
        """
        初始化音频处理器
        
        ⚠️ 当前使用 OpenAI Whisper API，等待用户确认 ASR 方案
        """
        self.provider = "openai"  # 临时方案
        
        # ASR 客户端
        if asr_client:
            self.asr_client = asr_client
        else:
            self.asr_client = self._create_asr_client(api_key)
        
        # 说话人分离配置
        self.enable_diarization = enable_diarization
        self.diarization_api_key = diarization_api_key or os.getenv('PYANNOTE_API_KEY')
    
    def _create_asr_client(self, api_key: Optional[str]) -> OpenAI:
        """创建 ASR 客户端（当前使用 OpenAI Whisper）"""
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI SDK 未安装。运行: pip install openai\n"
                "注意：当前使用 OpenAI Whisper API，因为 Bedrock 没有 ASR 模型。\n"
                "如需其他方案，请联系开发团队。"
            )
        
        key = api_key or os.getenv('OPENAI_API_KEY')
        if not key:
            raise ValueError(
                "OpenAI API Key 未配置。\n"
                "请设置环境变量 OPENAI_API_KEY\n\n"
                "说明：AWS Bedrock 目前没有 ASR 模型，\n"
                "我们临时使用 OpenAI Whisper API 进行语音转录。\n"
                "如需其他 ASR 方案（AWS Transcribe 等），请联系开发团队。"
            )
        
        return OpenAI(api_key=key)
    
    def supports(self, content_type: str) -> bool:
        """检查是否支持指定的内容类型"""
        return content_type.lower() in {'audio', 'voice', 'speech', 'recording'}
    
    def process(self, content: Any, metadata: dict = None) -> RawContent:
        """
        处理音频内容（使用 OpenAI Whisper API）
        
        ⚠️ 当前使用临时 ASR 方案，等待用户确认最终方案
        """
        metadata = metadata or {}
        
        try:
            # 获取音频数据
            audio_data, file_name = self._get_audio_data(content, metadata)
            
            # ASR 转录（当前使用 Whisper API）
            transcription = self._transcribe_with_api(audio_data, metadata)
            
            # 说话人分离（如果启用）
            if self.enable_diarization and self.diarization_api_key:
                transcription = self._apply_diarization_api(audio_data, transcription)
            
            # 格式化输出
            formatted_text = self._format_transcription(transcription)
            
            # 分块
            chunks = self._chunk_by_speaker_turns(transcription)
            
            # 摘要
            summary = self._generate_audio_summary(transcription)
            
            # 时间
            occurred_at = parse_date(metadata.get('recorded_date') or metadata.get('timestamp'))
            
            return RawContent(
                text=formatted_text,
                summary=summary,
                chunks=chunks,
                metadata={
                    'file_name': file_name,
                    'duration': transcription.get('duration'),
                    'language': transcription.get('language', 'unknown'),
                    'has_diarization': self.enable_diarization,
                    'asr_provider': self.provider,
                    'note': '使用 OpenAI Whisper API（临时方案，Bedrock 无 ASR 模型）',
                    **metadata
                },
                occurred_at=occurred_at
            )
            
        except Exception as e:
            logger.error(f"音频处理失败: {e}")
            return RawContent(
                text=f"[音频处理错误] {str(e)}",
                summary="音频处理失败",
                chunks=[],
                metadata={
                    'error': str(e),
                    'note': 'AWS Bedrock 无 ASR 模型，当前使用 OpenAI Whisper API',
                    **metadata
                },
                occurred_at=None
            )
    
    def _get_audio_data(self, content: Any, metadata: dict) -> tuple:
        """获取音频数据和文件名"""
        if isinstance(content, (str, Path)):
            audio_path = str(content)
            with open(audio_path, 'rb') as f:
                return f.read(), Path(audio_path).name
        elif isinstance(content, bytes):
            return content, metadata.get('file_name', 'audio.mp3')
        else:
            raise ValueError(f"不支持的音频内容类型: {type(content)}")
    
    def _transcribe_with_api(self, audio_data: bytes, metadata: dict) -> Dict:
        """使用 OpenAI Whisper API 进行转录"""
        language = metadata.get('language', 'zh')
        prompt = metadata.get('prompt')
        
        audio_file = io.BytesIO(audio_data)
        file_ext = self._detect_audio_format(audio_data)
        audio_file.name = f"audio.{file_ext}"
        
        response = self.asr_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language if language != 'auto' else None,
            prompt=prompt,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
        
        segments = []
        if hasattr(response, 'segments') and response.segments:
            for seg in response.segments:
                segments.append({
                    'start': seg.start,
                    'end': seg.end,
                    'text': seg.text.strip(),
                    'speaker': 'Speaker 1'
                })
        
        return {
            'text': response.text,
            'segments': segments,
            'language': response.language if hasattr(response, 'language') else language,
            'duration': segments[-1]['end'] if segments else 0
        }
    
    def _detect_audio_format(self, audio_data: bytes) -> str:
        """检测音频格式"""
        if audio_data[:3] == b'ID3' or audio_data[:2] == b'\xff\xfb':
            return 'mp3'
        elif audio_data[:4] == b'RIFF':
            return 'wav'
        elif audio_data[:4] == b'ftyp':
            return 'm4a'
        elif audio_data[:4] == b'OggS':
            return 'ogg'
        elif audio_data[:4] == b'fLaC':
            return 'flac'
        elif audio_data[:4] == b'\x1aE\xdf\xa3':
            return 'webm'
        else:
            return 'mp3'
    
    def _apply_diarization_api(self, audio_data: bytes, transcription: Dict) -> Dict:
        """使用 pyannote.ai API 进行说话人分离"""
        if not self.diarization_api_key:
            logger.warning("说话人分离未启用：缺少 PYANNOTE_API_KEY")
            return transcription
        
        try:
            import requests
            
            url = "https://api.pyannote.ai/v1/diarize"
            headers = {"Authorization": f"Bearer {self.diarization_api_key}"}
            files = {"file": ("audio.mp3", io.BytesIO(audio_data), "audio/mpeg")}
            
            response = requests.post(url, headers=headers, files=files)
            response.raise_for_status()
            
            diarization = response.json()
            
            segments = transcription.get('segments', [])
            for seg in segments:
                seg_start = seg.get('start', 0)
                for speaker_seg in diarization.get('segments', []):
                    if (speaker_seg['start'] <= seg_start < speaker_seg['end']):
                        seg['speaker'] = speaker_seg.get('speaker', 'Unknown')
                        break
            
            return transcription
            
        except Exception as e:
            logger.error(f"说话人分离 API 调用失败: {e}")
            return transcription
    
    def _format_transcription(self, transcription: Dict) -> str:
        """格式化转录结果"""
        segments = transcription.get('segments', [])
        
        if not segments:
            return transcription.get('text', '')
        
        lines = []
        for seg in segments:
            speaker = seg.get('speaker', 'Speaker 1')
            time_str = self._format_timestamp(seg.get('start', 0))
            text = seg.get('text', '').strip()
            
            if text:
                lines.append(f"[{speaker}] {time_str} {text}")
        
        return "\n".join(lines)
    
    def _format_timestamp(self, seconds: float) -> str:
        """格式化秒数为 MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def _chunk_by_speaker_turns(self, transcription: Dict) -> List[str]:
        """按说话人轮换分块"""
        segments = transcription.get('segments', [])
        
        if not segments:
            return []
        
        chunks = []
        current_chunk = []
        current_speaker = None
        
        for seg in segments:
            speaker = seg.get('speaker', 'Speaker 1')
            text = seg.get('text', '').strip()
            
            if not text:
                continue
            
            if speaker != current_speaker and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
            
            current_chunk.append(text)
            current_speaker = speaker
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [transcription.get('text', '')]
    
    def _generate_audio_summary(self, transcription: Dict) -> str:
        """生成音频摘要"""
        duration = transcription.get('duration', 0)
        text = transcription.get('text', '')
        language = transcription.get('language', 'unknown')
        
        parts = []
        
        if duration:
            mins = int(duration // 60)
            secs = int(duration % 60)
            parts.append(f"音频时长: {mins}分{secs}秒")
        
        parts.append(f"语言: {language}")
        
        if text:
            preview = text[:100].replace('\n', ' ')
            parts.append(f"预览: {preview}{'...' if len(text) > 100 else ''}")
        
        return " | ".join(parts)


# 便捷函数
def process_audio(
    audio_path: str,
    api_key: str = None,
    language: str = 'zh',
    enable_diarization: bool = False
) -> RawContent:
    """
    便捷函数：处理音频（使用 OpenAI Whisper API 临时方案）
    
    ⚠️ 注意：当前使用 OpenAI Whisper API，因为 AWS Bedrock 没有 ASR 模型。
    如需其他方案，请联系开发团队。
    """
    processor = AudioProcessor(
        api_key=api_key,
        enable_diarization=enable_diarization
    )
    return processor.process(audio_path, {
        'file_name': Path(audio_path).name,
        'language': language
    })
