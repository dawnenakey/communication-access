"""
GenASL Service - AWS GenAI ASL Avatar Integration

Integrates with AWS GenASL pipeline for realistic ASL avatar video generation.
Uses AWS Step Functions, S3, DynamoDB, and Bedrock for full sentence translation.

Reference: https://github.com/aws-samples/genai-asl-avatar-generator
"""

import boto3
import json
import logging
import os
import time
import uuid
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class GenASLStatus(str, Enum):
    """Status of GenASL video generation."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"


@dataclass
class GenASLVideoResult:
    """Result of GenASL video generation."""
    execution_id: str
    status: GenASLStatus
    video_url: Optional[str] = None
    gloss_sequence: Optional[List[str]] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    processing_time_ms: Optional[float] = None


class GenASLService:
    """
    Service for generating realistic ASL avatar videos using AWS GenASL pipeline.

    Architecture:
    - AWS Step Functions orchestrates the workflow
    - Amazon Transcribe converts speech to text (if audio input)
    - Amazon Bedrock (Claude) translates English to ASL gloss
    - DynamoDB stores gloss-to-video mappings from ASLLVD dataset
    - S3 hosts the pre-generated sign videos and output videos
    - Videos are stitched together for full sentence signing
    """

    # ASLLVD dataset has 3,300+ signs - much larger than the 50 we had before
    SUPPORTED_SIGNS_COUNT = 3300

    # Fingerspelling alphabet for fallback
    FINGERSPELLING = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def __init__(
        self,
        region: str = None,
        state_machine_arn: str = None,
        signs_table: str = None,
        videos_bucket: str = None
    ):
        """
        Initialize GenASL service.

        Args:
            region: AWS region (default from env)
            state_machine_arn: ARN of GenASL Step Functions state machine
            signs_table: DynamoDB table name for gloss-to-video mappings
            videos_bucket: S3 bucket for avatar videos
        """
        self.region = region or os.environ.get('AWS_REGION', 'us-east-1')
        self.state_machine_arn = state_machine_arn or os.environ.get('GENASL_STATE_MACHINE_ARN')
        self.signs_table = signs_table or os.environ.get('GENASL_SIGNS_TABLE', 'genasl-signs')
        self.videos_bucket = videos_bucket or os.environ.get('GENASL_VIDEOS_BUCKET', 'genasl-videos')

        # Initialize AWS clients
        self.sfn_client = None
        self.dynamodb = None
        self.s3_client = None
        self.bedrock_client = None

        self._initialize_clients()

        # Cache for available signs (loaded from DynamoDB)
        self._available_signs_cache: Optional[Dict[str, Any]] = None
        self._cache_timestamp: float = 0
        self._cache_ttl: float = 3600  # 1 hour cache

    def _initialize_clients(self):
        """Initialize AWS clients."""
        try:
            self.sfn_client = boto3.client('stepfunctions', region_name=self.region)
            self.dynamodb = boto3.resource('dynamodb', region_name=self.region)
            self.s3_client = boto3.client('s3', region_name=self.region)
            self.bedrock_client = boto3.client('bedrock-runtime', region_name=self.region)
            logger.info(f"GenASL service initialized (region: {self.region})")
        except Exception as e:
            logger.error(f"Failed to initialize GenASL AWS clients: {e}")

    def is_available(self) -> bool:
        """Check if GenASL service is properly configured."""
        return all([
            self.sfn_client is not None,
            self.state_machine_arn is not None,
            self.videos_bucket is not None
        ])

    async def get_available_signs(self) -> Dict[str, Any]:
        """
        Get all available ASL signs from the ASLLVD dataset.

        Returns dict with:
        - signs: List of all available gloss strings
        - count: Total number of available signs
        - categories: Signs organized by category
        """
        # Check cache
        if (self._available_signs_cache and
            time.time() - self._cache_timestamp < self._cache_ttl):
            return self._available_signs_cache

        try:
            table = self.dynamodb.Table(self.signs_table)

            # Scan for all available signs
            signs = []
            categories = {}

            response = table.scan(
                ProjectionExpression='gloss, category, video_key'
            )

            for item in response.get('Items', []):
                gloss = item.get('gloss')
                if gloss:
                    signs.append(gloss)
                    category = item.get('category', 'general')
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(gloss)

            # Handle pagination
            while 'LastEvaluatedKey' in response:
                response = table.scan(
                    ProjectionExpression='gloss, category, video_key',
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                for item in response.get('Items', []):
                    gloss = item.get('gloss')
                    if gloss:
                        signs.append(gloss)
                        category = item.get('category', 'general')
                        if category not in categories:
                            categories[category] = []
                        categories[category].append(gloss)

            result = {
                'signs': sorted(signs),
                'count': len(signs),
                'categories': categories,
                'source': 'ASLLVD',
                'fingerspelling_available': True
            }

            # Update cache
            self._available_signs_cache = result
            self._cache_timestamp = time.time()

            return result

        except Exception as e:
            logger.error(f"Error fetching available signs: {e}")
            # Return fallback with common signs
            return self._get_fallback_signs()

    def _get_fallback_signs(self) -> Dict[str, Any]:
        """Return fallback sign list if DynamoDB is unavailable."""
        common_signs = [
            # Greetings & Social
            "HELLO", "GOODBYE", "NICE-TO-MEET-YOU", "THANK-YOU", "PLEASE",
            "SORRY", "EXCUSE-ME", "WELCOME", "CONGRATULATIONS",

            # Questions
            "WHAT", "WHERE", "WHO", "WHY", "HOW", "WHEN", "WHICH", "HOW-MUCH",
            "HOW-MANY", "HOW-OLD",

            # Pronouns & People
            "I", "ME", "YOU", "HE", "SHE", "IT", "WE", "THEY", "MY", "YOUR",
            "HIS", "HER", "OUR", "THEIR", "MYSELF", "YOURSELF",

            # Common Verbs
            "WANT", "NEED", "LIKE", "LOVE", "HATE", "KNOW", "UNDERSTAND",
            "THINK", "BELIEVE", "REMEMBER", "FORGET", "LEARN", "TEACH",
            "HELP", "GIVE", "TAKE", "GET", "HAVE", "MAKE", "DO", "GO",
            "COME", "SEE", "LOOK", "WATCH", "HEAR", "LISTEN", "SAY", "TELL",
            "ASK", "ANSWER", "CALL", "WAIT", "STOP", "START", "FINISH",
            "TRY", "WORK", "PLAY", "READ", "WRITE", "EAT", "DRINK", "SLEEP",
            "WAKE-UP", "SIT", "STAND", "WALK", "RUN", "DRIVE", "FLY",
            "BUY", "SELL", "PAY", "COST", "LIVE", "DIE", "BORN",

            # Adjectives & Feelings
            "GOOD", "BAD", "HAPPY", "SAD", "ANGRY", "SCARED", "SURPRISED",
            "TIRED", "SICK", "FINE", "OKAY", "BEAUTIFUL", "UGLY", "BIG",
            "SMALL", "TALL", "SHORT", "LONG", "NEW", "OLD", "YOUNG",
            "HOT", "COLD", "WARM", "COOL", "FAST", "SLOW", "EASY", "HARD",
            "RIGHT", "WRONG", "TRUE", "FALSE", "SAME", "DIFFERENT",
            "IMPORTANT", "INTERESTING", "BORING", "FUNNY", "SERIOUS",

            # Time
            "NOW", "TODAY", "TOMORROW", "YESTERDAY", "WEEK", "MONTH", "YEAR",
            "MORNING", "AFTERNOON", "EVENING", "NIGHT", "EARLY", "LATE",
            "BEFORE", "AFTER", "ALWAYS", "NEVER", "SOMETIMES", "OFTEN",
            "AGAIN", "STILL", "ALREADY", "YET", "SOON", "LATER",

            # Numbers
            "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT",
            "NINE", "TEN", "HUNDRED", "THOUSAND", "MILLION", "FIRST",
            "SECOND", "THIRD", "LAST", "NEXT", "FEW", "MANY", "MORE", "LESS",
            "ALL", "SOME", "NONE", "BOTH", "EACH", "EVERY", "OTHER",

            # Places & Things
            "HOME", "HOUSE", "SCHOOL", "WORK", "STORE", "HOSPITAL", "CHURCH",
            "RESTAURANT", "LIBRARY", "PARK", "CITY", "COUNTRY", "WORLD",
            "ROOM", "DOOR", "WINDOW", "TABLE", "CHAIR", "BED", "FOOD",
            "WATER", "MONEY", "CAR", "PHONE", "COMPUTER", "BOOK", "PAPER",

            # Family
            "FAMILY", "MOTHER", "FATHER", "PARENTS", "SISTER", "BROTHER",
            "SON", "DAUGHTER", "BABY", "CHILDREN", "GRANDMOTHER", "GRANDFATHER",
            "AUNT", "UNCLE", "COUSIN", "FRIEND", "HUSBAND", "WIFE",

            # Colors
            "RED", "BLUE", "GREEN", "YELLOW", "ORANGE", "PURPLE", "PINK",
            "BLACK", "WHITE", "BROWN", "GRAY",

            # Other Common
            "YES", "NO", "MAYBE", "NOT", "AND", "OR", "BUT", "BECAUSE",
            "IF", "THEN", "FOR", "WITH", "WITHOUT", "ABOUT", "VERY",
            "REALLY", "JUST", "ONLY", "ALSO", "TOO", "ENOUGH", "THING",
            "SOMETHING", "NOTHING", "EVERYTHING", "SOMEONE", "ANYONE",
            "EVERYONE", "NOBODY", "HERE", "THERE", "ANYWHERE", "EVERYWHERE"
        ]

        return {
            'signs': common_signs,
            'count': len(common_signs),
            'categories': {
                'greetings': common_signs[:9],
                'questions': common_signs[9:19],
                'pronouns': common_signs[19:35],
                'verbs': common_signs[35:95],
                'adjectives': common_signs[95:140],
                'time': common_signs[140:166],
                'numbers': common_signs[166:198],
                'places': common_signs[198:226],
                'family': common_signs[226:244],
                'colors': common_signs[244:255],
                'common': common_signs[255:]
            },
            'source': 'fallback',
            'fingerspelling_available': True
        }

    async def translate_to_gloss(self, english_text: str) -> List[str]:
        """
        Translate English text to ASL gloss sequence using Bedrock.

        Uses Claude to intelligently translate following ASL grammar rules:
        - Topic-comment structure
        - Time-first ordering
        - Removal of articles and auxiliary verbs
        - Proper ASL vocabulary

        Returns list of gloss strings that can be looked up in ASLLVD.
        """
        if not self.bedrock_client:
            return self._simple_gloss(english_text)

        # Get available signs for better translation
        available = await self.get_available_signs()
        sign_list = available.get('signs', [])[:500]  # Include subset for context

        prompt = f"""You are an expert English to ASL (American Sign Language) gloss translator.

Convert the following English text into ASL gloss format for a signing avatar.

ASL GRAMMAR RULES:
1. Use CAPITAL-LETTERS with hyphens for multi-word concepts (e.g., NICE-TO-MEET-YOU)
2. Topic-comment structure: Put the topic first, then comment
3. Time concepts come first (TOMORROW I WORK, not I WILL WORK TOMORROW)
4. Remove articles (a, an, the)
5. Remove auxiliary/helper verbs (is, are, was, were, will, would, could)
6. Use ASL vocabulary - prefer single sign concepts over word-for-word
7. Questions end with the question word (YOU NAME WHAT, not WHAT IS YOUR NAME)

AVAILABLE SIGNS IN OUR DATABASE (prefer these):
{', '.join(sign_list[:200])}
... and {len(sign_list) - 200} more signs available

If a word doesn't have a sign, use #WORD format for fingerspelling (e.g., #JOHN for names)

English: {english_text}

Respond with ONLY the ASL gloss sequence, one sign per word, separated by spaces.
Example: HELLO MY NAME #JOHN NICE-TO-MEET-YOU"""

        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}]
            })

            response = self.bedrock_client.invoke_model(
                modelId="anthropic.claude-3-sonnet-20240229-v1:0",  # Use Sonnet for better translation
                body=body,
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response['body'].read())
            gloss_text = response_body['content'][0]['text'].strip()

            # Parse gloss sequence
            glosses = gloss_text.split()

            # Clean up and validate
            validated_glosses = []
            for gloss in glosses:
                # Remove any quotes or extra punctuation
                gloss = gloss.strip('"\'.,!?')
                if gloss:
                    validated_glosses.append(gloss.upper())

            return validated_glosses

        except Exception as e:
            logger.error(f"Bedrock translation error: {e}")
            return self._simple_gloss(english_text)

    def _simple_gloss(self, text: str) -> List[str]:
        """Simple fallback gloss conversion."""
        import re
        words = re.sub(r'[^\w\s]', '', text).upper().split()
        skip_words = {'A', 'AN', 'THE', 'IS', 'ARE', 'WAS', 'WERE', 'AM', 'BE',
                      'BEEN', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'TO', 'OF'}
        return [w for w in words if w not in skip_words]

    async def lookup_sign_video(self, gloss: str) -> Optional[Dict[str, Any]]:
        """
        Look up the video for a specific sign gloss in DynamoDB.

        Returns dict with video_key, duration, and presigned URL if found.
        """
        try:
            table = self.dynamodb.Table(self.signs_table)

            # Handle fingerspelling
            if gloss.startswith('#'):
                # Return fingerspelling sequence
                letters = gloss[1:]
                return {
                    'gloss': gloss,
                    'type': 'fingerspelling',
                    'letters': list(letters),
                    'video_keys': [f"fingerspelling/{letter}.mp4" for letter in letters]
                }

            response = table.get_item(Key={'gloss': gloss})

            if 'Item' in response:
                item = response['Item']
                video_key = item.get('video_key')

                # Generate presigned URL
                if video_key:
                    url = self.s3_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': self.videos_bucket, 'Key': video_key},
                        ExpiresIn=3600  # 1 hour
                    )
                    return {
                        'gloss': gloss,
                        'video_key': video_key,
                        'video_url': url,
                        'duration': item.get('duration', 1.0),
                        'category': item.get('category', 'general')
                    }

            return None

        except Exception as e:
            logger.error(f"Error looking up sign {gloss}: {e}")
            return None

    async def generate_sentence_video(
        self,
        english_text: str,
        audio_input: Optional[bytes] = None,
        execution_wait: bool = True,
        max_wait_seconds: int = 60
    ) -> GenASLVideoResult:
        """
        Generate a complete ASL sentence video from English text or audio.

        This triggers the AWS Step Functions state machine which:
        1. Transcribes audio (if provided)
        2. Translates English to ASL gloss via Bedrock
        3. Looks up each sign video from DynamoDB
        4. Stitches videos together
        5. Returns presigned S3 URL for the final video

        Args:
            english_text: English text to translate and sign
            audio_input: Optional audio bytes (will be transcribed first)
            execution_wait: If True, wait for execution to complete
            max_wait_seconds: Maximum time to wait for completion

        Returns:
            GenASLVideoResult with video URL and metadata
        """
        if not self.is_available():
            return GenASLVideoResult(
                execution_id="",
                status=GenASLStatus.FAILED,
                error_message="GenASL service not configured. Set GENASL_STATE_MACHINE_ARN."
            )

        start_time = time.time()
        execution_id = f"genasl-{uuid.uuid4().hex[:8]}"

        try:
            # Prepare input for Step Functions
            sfn_input = {
                "execution_id": execution_id,
                "input_type": "audio" if audio_input else "text",
                "english_text": english_text,
                "timestamp": time.time()
            }

            # If audio, upload to S3 first
            if audio_input:
                audio_key = f"inputs/{execution_id}/audio.wav"
                self.s3_client.put_object(
                    Bucket=self.videos_bucket,
                    Key=audio_key,
                    Body=audio_input,
                    ContentType='audio/wav'
                )
                sfn_input["audio_s3_key"] = audio_key

            # Start Step Functions execution
            response = self.sfn_client.start_execution(
                stateMachineArn=self.state_machine_arn,
                name=execution_id,
                input=json.dumps(sfn_input)
            )

            execution_arn = response['executionArn']

            if not execution_wait:
                return GenASLVideoResult(
                    execution_id=execution_id,
                    status=GenASLStatus.RUNNING,
                    processing_time_ms=(time.time() - start_time) * 1000
                )

            # Poll for completion
            while time.time() - start_time < max_wait_seconds:
                status_response = self.sfn_client.describe_execution(
                    executionArn=execution_arn
                )

                status = status_response['status']

                if status == 'SUCCEEDED':
                    output = json.loads(status_response.get('output', '{}'))
                    video_url = output.get('video_url')

                    # Generate presigned URL if we got an S3 key
                    if not video_url and output.get('video_s3_key'):
                        video_url = self.s3_client.generate_presigned_url(
                            'get_object',
                            Params={
                                'Bucket': self.videos_bucket,
                                'Key': output['video_s3_key']
                            },
                            ExpiresIn=3600
                        )

                    return GenASLVideoResult(
                        execution_id=execution_id,
                        status=GenASLStatus.SUCCEEDED,
                        video_url=video_url,
                        gloss_sequence=output.get('gloss_sequence', []),
                        duration_seconds=output.get('duration_seconds'),
                        processing_time_ms=(time.time() - start_time) * 1000
                    )

                elif status == 'FAILED':
                    error = status_response.get('error', 'Unknown error')
                    cause = status_response.get('cause', '')
                    return GenASLVideoResult(
                        execution_id=execution_id,
                        status=GenASLStatus.FAILED,
                        error_message=f"{error}: {cause}",
                        processing_time_ms=(time.time() - start_time) * 1000
                    )

                elif status in ['TIMED_OUT', 'ABORTED']:
                    return GenASLVideoResult(
                        execution_id=execution_id,
                        status=GenASLStatus.TIMED_OUT,
                        error_message=f"Execution {status.lower()}",
                        processing_time_ms=(time.time() - start_time) * 1000
                    )

                # Still running, wait and poll again
                time.sleep(1)

            # Timeout waiting for completion
            return GenASLVideoResult(
                execution_id=execution_id,
                status=GenASLStatus.RUNNING,
                error_message="Timeout waiting for completion - video may still be generating",
                processing_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            logger.error(f"GenASL video generation error: {e}")
            return GenASLVideoResult(
                execution_id=execution_id,
                status=GenASLStatus.FAILED,
                error_message=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )

    async def get_execution_status(self, execution_id: str) -> GenASLVideoResult:
        """
        Check the status of a video generation execution.

        Use this to poll for async execution results.
        """
        try:
            # List recent executions to find the one with matching name
            response = self.sfn_client.list_executions(
                stateMachineArn=self.state_machine_arn,
                maxResults=100
            )

            for execution in response.get('executions', []):
                if execution['name'] == execution_id:
                    execution_arn = execution['executionArn']

                    status_response = self.sfn_client.describe_execution(
                        executionArn=execution_arn
                    )

                    status_map = {
                        'RUNNING': GenASLStatus.RUNNING,
                        'SUCCEEDED': GenASLStatus.SUCCEEDED,
                        'FAILED': GenASLStatus.FAILED,
                        'TIMED_OUT': GenASLStatus.TIMED_OUT,
                        'ABORTED': GenASLStatus.FAILED
                    }

                    status = status_map.get(status_response['status'], GenASLStatus.FAILED)

                    result = GenASLVideoResult(
                        execution_id=execution_id,
                        status=status
                    )

                    if status == GenASLStatus.SUCCEEDED:
                        output = json.loads(status_response.get('output', '{}'))
                        result.video_url = output.get('video_url')
                        result.gloss_sequence = output.get('gloss_sequence')
                        result.duration_seconds = output.get('duration_seconds')

                        if not result.video_url and output.get('video_s3_key'):
                            result.video_url = self.s3_client.generate_presigned_url(
                                'get_object',
                                Params={
                                    'Bucket': self.videos_bucket,
                                    'Key': output['video_s3_key']
                                },
                                ExpiresIn=3600
                            )
                    elif status == GenASLStatus.FAILED:
                        result.error_message = status_response.get('error', 'Unknown error')

                    return result

            return GenASLVideoResult(
                execution_id=execution_id,
                status=GenASLStatus.FAILED,
                error_message="Execution not found"
            )

        except Exception as e:
            logger.error(f"Error checking execution status: {e}")
            return GenASLVideoResult(
                execution_id=execution_id,
                status=GenASLStatus.FAILED,
                error_message=str(e)
            )

    async def health_check(self) -> Dict[str, Any]:
        """Check GenASL service health and configuration."""
        health = {
            "service": "genasl",
            "configured": self.is_available(),
            "region": self.region,
            "state_machine_arn": self.state_machine_arn,
            "videos_bucket": self.videos_bucket,
            "signs_table": self.signs_table,
            "clients": {
                "step_functions": self.sfn_client is not None,
                "dynamodb": self.dynamodb is not None,
                "s3": self.s3_client is not None,
                "bedrock": self.bedrock_client is not None
            }
        }

        # Test DynamoDB connection
        if self.dynamodb:
            try:
                table = self.dynamodb.Table(self.signs_table)
                response = table.scan(Limit=1)
                health["dynamodb_connected"] = True
                health["sample_sign"] = response.get('Items', [{}])[0].get('gloss') if response.get('Items') else None
            except Exception as e:
                health["dynamodb_connected"] = False
                health["dynamodb_error"] = str(e)

        # Test S3 connection
        if self.s3_client:
            try:
                self.s3_client.head_bucket(Bucket=self.videos_bucket)
                health["s3_connected"] = True
            except Exception as e:
                health["s3_connected"] = False
                health["s3_error"] = str(e)

        return health


# Global singleton instance
_genasl_service: Optional[GenASLService] = None


def get_genasl_service() -> GenASLService:
    """Get or create the global GenASL service instance."""
    global _genasl_service
    if _genasl_service is None:
        _genasl_service = GenASLService()
    return _genasl_service
