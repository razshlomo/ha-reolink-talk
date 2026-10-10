"""Proves media_player entities come from the Reolink core integration's own
connected Host: one entity per hub channel, named after the camera, and no
second Host or Baichuan session opened for setup, ability probe or playback.

Regression: a Home Hub on HTTPS (self-signed certificate) made the old probe
Host fail silently, which fell back to a single entity on channel 0 named after
the config entry ("hub") -- a camera with no speaker, not the doorbell.
"""
import importlib.util
import pathlib
import sys
import types
import unittest


def _stub(name, **attrs):
    mod = sys.modules.setdefault(name, types.ModuleType(name))
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


class _Feature:
    PLAY_MEDIA = VOLUME_SET = BROWSE_MEDIA = 1


class _State:
    IDLE = "idle"
    PLAYING = "playing"


class _MediaPlayerEntity:
    def async_write_ha_state(self):
        pass


_stub("homeassistant")
_stub("homeassistant.components")
_stub("homeassistant.components.media_player", MediaPlayerEntity=_MediaPlayerEntity)
_stub("homeassistant.components.media_player.const",
      MediaPlayerEntityFeature=_Feature, MediaType=object, MediaPlayerState=_State)
_stub("homeassistant.config_entries", ConfigEntry=object)
_stub("homeassistant.core", HomeAssistant=object, callback=lambda f: f)
_stub("homeassistant.helpers")
_stub("homeassistant.helpers.entity_platform", AddEntitiesCallback=object)
_stub("homeassistant.helpers.aiohttp_client", async_get_clientsession=None)
_stub("homeassistant.helpers.network", get_url=None)


class _NoSecondHost:
    def __init__(self, *a, **kw):
        raise AssertionError("must reuse the Reolink core integration's Host")


_stub("reolink_aio")
_stub("reolink_aio.api", Host=_NoSecondHost)

_pkg_dir = pathlib.Path(__file__).parent.parent / "custom_components/reolink_talk"
_pkg = types.ModuleType("reolink_talk_mp_under_test")
_pkg.__path__ = [str(_pkg_dir)]
sys.modules["reolink_talk_mp_under_test"] = _pkg
_spec = importlib.util.spec_from_file_location(
    "reolink_talk_mp_under_test.media_player", _pkg_dir / "media_player.py"
)
mp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mp
_spec.loader.exec_module(mp)

ABILITY_XML = (
    "<body><TalkAbility><duplexList><duplex>FDX</duplex></duplexList>"
    "<audioStreamModeList><audioStreamMode>mixAudioStream</audioStreamMode></audioStreamModeList>"
    "<audioConfigList><audioConfig><audioType>adpcm</audioType><sampleRate>16000</sampleRate>"
    "<lengthPerEncoder>1024</lengthPerEncoder></audioConfig></audioConfigList>"
    "</TalkAbility></body>"
)


class FakeBaichuan:
    def __init__(self):
        self.sent = []

    async def login(self):
        raise AssertionError("the core integration's session is already logged in")

    async def send(self, cmd_id, channel=None, **_kw):
        self.sent.append((cmd_id, channel))
        return ABILITY_XML


class FakeApi:
    def __init__(self, names):
        self._names = names
        self.channels = list(names)
        self.baichuan = FakeBaichuan()

    def camera_name(self, ch):
        return self._names[ch]


def _entry(entry_id, title, api):
    runtime = types.SimpleNamespace(host=types.SimpleNamespace(api=api)) if api else None
    data = {"host": "192.168.1.75", "username": "u", "password": "p", "use_https": True, "port": 443}
    return types.SimpleNamespace(entry_id=entry_id, title=title, data=data, runtime_data=runtime)


class FakeConfigEntries:
    def __init__(self, entries):
        self._entries = {e.entry_id: e for e in entries}

    def async_entries(self, domain):
        return list(self._entries.values())

    def async_get_entry(self, entry_id):
        return self._entries.get(entry_id)


class MediaPlayerChannelsTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.api = FakeApi({0: "tinok", 1: "Front-door", 5: "games"})
        self.hass = types.SimpleNamespace(config_entries=FakeConfigEntries([
            _entry("hub_id", "hub", self.api),
            _entry("old_cam_id", "yard (192.168.1.73)", None),  # not loaded
        ]))
        self.added = []

    async def _setup(self, options=None):
        entry = types.SimpleNamespace(options=options or {})
        await mp.async_setup_entry(self.hass, entry, lambda ents, update_before_add=False: self.added.extend(ents))

    async def test_one_entity_per_hub_channel_named_after_camera(self):
        await self._setup()
        got = sorted((e._target.channel, e._attr_name, e._attr_unique_id) for e in self.added)
        self.assertEqual(got, [
            (0, "Reolink Talk tinok", "reolink_talk:hub_id:0"),
            (1, "Reolink Talk Front-door", "reolink_talk:hub_id:1"),
            (5, "Reolink Talk games", "reolink_talk:hub_id:5"),
        ])

    async def test_unloaded_entry_gets_no_guessed_entity(self):
        await self._setup({"reolink_entry_ids": ["old_cam_id"]})
        self.assertEqual(self.added, [])

    async def test_probe_and_playback_use_core_baichuan_on_right_channel(self):
        await self._setup()
        doorbell = next(e for e in self.added if e._target.channel == 1)
        played = []

        async def fake_ffmpeg(media_bytes, sample_rate, volume):
            return b"\x00\x00" * 2048

        async def fake_talk(bc, channel, adpcm, ability, block_align):
            played.append((bc, channel, block_align))

        mp.ffmpeg_to_pcm_s16le = fake_ffmpeg
        mp.talk_playback = fake_talk
        await doorbell._play_bytes(b"audio")

        self.assertEqual(self.api.baichuan.sent, [(10, 1)])
        self.assertEqual(played, [(self.api.baichuan, 1, 516)])

    async def test_playback_after_core_entry_unloads_raises(self):
        await self._setup()
        doorbell = next(e for e in self.added if e._target.channel == 1)
        self.hass.config_entries._entries["hub_id"].runtime_data = None
        with self.assertRaises(RuntimeError):
            await doorbell._play_bytes(b"audio")


if __name__ == "__main__":
    unittest.main()
