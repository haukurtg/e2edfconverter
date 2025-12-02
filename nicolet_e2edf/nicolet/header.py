"""Header parsing utilities for Nicolet `.e` recordings."""

# This file includes logic adapted from FieldTrip's read_nervus_header.m.
# FieldTrip is released under the GPL-3.0 licence. Copyright (C) the FieldTrip project.

from __future__ import annotations

import struct
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .types import EventItem, MainIndexEntry, NervusHeader, SegmentInfo, StaticPacket, TSEntry

# ---------------------------------------------------------------------------
# Constants and simple helpers
# ---------------------------------------------------------------------------

DAY_SECONDS = 86400.0
DATETIME_MINUS_FACTOR = 2_209_161_600
POSIX_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _safe_posix_datetime(seconds: float) -> datetime:
    """Return a timezone-aware datetime, clamping out-of-range values."""

    try:
        return POSIX_EPOCH + timedelta(seconds=seconds)
    except (OverflowError, OSError, ValueError):
        return POSIX_EPOCH


STATIC_PACKET_ID_MAP = {
    "ExtraDataStaticPackets": "ExtraDataStaticPackets",
    "SegmentStream": "SegmentStream",
    "DataStream": "DataStream",
    "ExtraDataTags": "ExtraDataTags",
    "InfoChangeStream": "InfoChangeStream",
    "InfoGuids": "InfoGuids",
    "{A271CCCB-515D-4590-B6A1-DC170C8D6EE2}": "TSGUID",
    "{8A19AA48-BEA0-40D5-B89F-667FC578D635}": "DERIVATIONGUID",
    "{F824D60C-995E-4D94-9578-893C755ECB99}": "FILTERGUID",
    "{02950361-35BB-4A22-9F0B-C78AAA5DB094}": "DISPLAYGUID",
    "{8E9421-70F5-11D3-8F72-00105A9AFD56}": "FILEINFOGUID",
    "{E4138BC0-7733-11D3-8685-0050044DAAB1}": "SRINFOGUID",
    "{C728E565-E5A0-4419-93D2-F6CFC69F3B8F}": "EVENTTYPEINFOGUID",
    "{D01B34A0-9DBD-11D3-93D3-00500400C148}": "AUDIOINFOGUID",
    "{BF7C95EF-6C3B-4E70-9E11-779BFFF58EA7}": "CHANNELGUID",
    "{2DEB82A1-D15F-4770-A4A4-CF03815F52DE}": "INPUTGUID",
    "{5B036022-2EDC-465F-86EC-C0A4AB1A7A91}": "INPUTSETTINGSGUID",
    "{99A636F2-51F7-4B9D-9569-C7D45058431A}": "PHOTICGUID",
    "{55C5E044-5541-4594-9E35-5B3004EF7647}": "ERRORGUID",
    "{223A3CA0-B5AC-43FB-B0A8-74CF8752BDBE}": "VIDEOGUID",
    "{0623B545-38BE-4939-B9D0-55F5E241278D}": "DETECTIONPARAMSGUID",
    "{CE06297D-D9D6-4E4B-8EAC-305EA1243EAB}": "PAGEGUID",
    "{782B34E8-8E51-4BB9-9701-3227BB882A23}": "ACCINFOGUID",
    "{3A6E8546-D144-4B55-A2C7-40DF579ED11E}": "RECCTRLGUID",
    "{D046F2B0-5130-41B1-ABD7-38C12B32FAC3}": "GUID TRENDINFOGUID",
    "{CBEBA8E6-1CDA-4509-B6C2-6AC2EA7DB8F8}": "HWINFOGUID",
    "{E11C4CBA-0753-4655-A1E9-2B2309D1545B}": "VIDEOSYNCGUID",
    "{B9344241-7AC1-42B5-BE9B-B7AFA16CBFA5}": "SLEEPSCOREINFOGUID",
    "{15B41C32-0294-440E-ADFF-DD8B61C8B5AE}": "FOURIERSETTINGSGUID",
    "{024FA81F-6A83-43C8-8C82-241A5501F0A1}": "SPECTRUMGUID",
    "{8032E68A-EA3E-42E8-893E-6E93C59ED515}": "SIGNALINFOGUID",
    "{30950D98-C39C-4352-AF3E-CB17D5B93DED}": "SENSORINFOGUID",
    "{F5D39CD3-A340-4172-A1A3-78B2CDBCCB9F}": "DERIVEDSIGNALINFOGUID",
    "{969FBB89-EE8E-4501-AD40-FB5A448BC4F9}": "ARTIFACTINFOGUID",
    "{02948284-17EC-4538-A7FA-8E18BD65E167}": "STUDYINFOGUID",
    "{D0B3FD0B-49D9-4BF0-8929-296DE5A55910}": "PATIENTINFOGUID",
    "{7842FEF5-A686-459D-8196-769FC0AD99B3}": "DOCUMENTINFOGUID",
    "{BCDAEE87-2496-4DF4-B07C-8B4E31E3C495}": "USERSINFOGUID",
    "{B799F680-72A4-11D3-93D3-00500400C148}": "EVENTGUID",
    "{AF2B3281-7FCE-11D2-B2DE-00104B6FC652}": "SHORTSAMPLESGUID",
    "{89A091B3-972E-4DA2-9266-261B186302A9}": "DELAYLINESAMPLESGUID",
    "{291E2381-B3B4-44D1-BB77-8CF5C24420D7}": "GENERALSAMPLESGUID",
    "{5F11C628-FCCC-4FDD-B429-5EC94CB3AFEB}": "FILTERSAMPLESGUID",
    "{728087F8-73E1-44D1-8882-C770976478A2}": "DATEXDATAGUID",
    "{35F356D9-0F1C-4DFE-8286-D3DB3346FD75}": "TESTINFOGUID",
}

_UINT16 = struct.Struct("<H")
_UINT32 = struct.Struct("<I")
_UINT64 = struct.Struct("<Q")
_DOUBLE = struct.Struct("<d")


# ---------------------------------------------------------------------------
# Binary readers
# ---------------------------------------------------------------------------


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise EOFError(f"Unexpected end of file while reading {size} bytes")
    return data


def _read_u16(handle: BinaryIO) -> int:
    return _UINT16.unpack(_read_exact(handle, _UINT16.size))[0]


def _read_u32(handle: BinaryIO) -> int:
    return _UINT32.unpack(_read_exact(handle, _UINT32.size))[0]


def _read_u64(handle: BinaryIO) -> int:
    return _UINT64.unpack(_read_exact(handle, _UINT64.size))[0]


def _read_double(handle: BinaryIO) -> float:
    return _DOUBLE.unpack(_read_exact(handle, _DOUBLE.size))[0]


def _read_utf16(handle: BinaryIO, code_units: int) -> str:
    raw = _read_exact(handle, code_units * 2)
    return raw.decode("utf-16le", errors="ignore").split("\x00", 1)[0].strip()


def _decode_utf16(data: bytes) -> str:
    return data.decode("utf-16le", errors="ignore").split("\x00", 1)[0].strip()


def _mixed_endian_guid(raw: bytes) -> tuple[str, str]:
    if len(raw) != 16:
        raise ValueError("GUID payload must be exactly 16 bytes")
    part1 = raw[3::-1]
    part2 = raw[5:3:-1]
    part3 = raw[7:5:-1]
    remainder = raw[8:]
    compact = (part1 + part2 + part3 + remainder).hex().upper()
    pretty = (
        f"{{{compact[0:8]}-{compact[8:12]}-{compact[12:16]}-"
        f"{compact[16:20]}-{compact[20:32]}}}"
    )
    return compact, pretty


def _resolve_static_packet_id(tag: str) -> str:
    cleaned = tag.rstrip()
    if cleaned in STATIC_PACKET_ID_MAP:
        return STATIC_PACKET_ID_MAP[cleaned]
    if cleaned.isdigit():
        return cleaned
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Low-level structure loading
# ---------------------------------------------------------------------------


def _read_static_packets(handle: BinaryIO) -> list[StaticPacket]:
    handle.seek(172, 0)
    count = _read_u32(handle)
    packets: list[StaticPacket] = []
    for _ in range(count):
        tag = _read_utf16(handle, 40)
        index = _read_u32(handle)
        packets.append(StaticPacket(tag=tag, index=index, IDStr=_resolve_static_packet_id(tag)))
    return packets


def _read_qi_index(handle: BinaryIO, nr_static_packets: int) -> dict[str, object]:
    handle.seek(172_208, 0)
    return {
        "nrEntries": _read_u32(handle),
        "misc1": _read_u32(handle),
        "indexIdx": _read_u32(handle),
        "misc3": _read_u32(handle),
        "LQi": _read_u64(handle),
        "firstIdx": [_read_u64(handle) for _ in range(nr_static_packets)],
    }


def _read_qi_index2(handle: BinaryIO, qi_index: dict[str, object]) -> list[dict[str, object]]:
    handle.seek(188_664, 0)
    lqi = int(qi_index.get("LQi", 0) or 0)
    entries: list[dict[str, object]] = []
    for _ in range(lqi):
        index_low = _read_u16(handle)
        index_high = _read_u16(handle)
        misc1 = _read_u32(handle)
        index_idx = _read_u32(handle)
        misc2 = [_read_u32(handle) for _ in range(3)]
        section_idx = _read_u32(handle)
        misc3 = _read_u32(handle)
        offset = _read_u64(handle)
        block_and_section = _read_u64(handle)
        block_len = block_and_section & 0xFFFFFFFF
        section_len = (block_and_section >> 32) & 0xFFFFFFFF
        data_len = _read_u32(handle)
        entries.append(
            {
                "index": (index_low, index_high),
                "misc1": misc1,
                "indexIdx": index_idx,
                "misc2": misc2,
                "sectionIdx": section_idx,
                "misc3": misc3,
                "offset": offset,
                "blockL": block_len,
                "sectionL": section_len,
                "dataL": data_len,
            }
        )
    return entries


def _read_main_index(handle: BinaryIO, index_idx: int, nr_entries: int) -> list[MainIndexEntry]:
    entries: list[MainIndexEntry] = []
    next_pointer = index_idx
    read_entries = 0
    while read_entries < nr_entries:
        handle.seek(next_pointer, 0)
        nr_idx = _read_u64(handle)
        chunk = [_read_u64(handle) for _ in range(3 * nr_idx)]
        for i in range(int(nr_idx)):
            section_idx = chunk[3 * i]
            offset = chunk[3 * i + 1]
            block_l_raw = chunk[3 * i + 2]
            block_len = block_l_raw & 0xFFFFFFFF
            section_len = (block_l_raw >> 32) & 0xFFFFFFFF
            entries.append(
                MainIndexEntry(
                    sectionIdx=int(section_idx),
                    offset=int(offset),
                    blockL=int(block_len),
                    sectionL=int(section_len),
                )
            )
        next_pointer = _read_u64(handle)
        read_entries += int(nr_idx)
    return entries


def _lookup_static(
    static_packets: Iterable[StaticPacket],
    *,
    idstr: str | None = None,
    tag: str | None = None,
) -> StaticPacket | None:
    for packet in static_packets:
        if idstr is not None and packet.IDStr == idstr:
            return packet
        if tag is not None and packet.tag == tag:
            return packet
    return None


def _main_index_by_position(main_index: Sequence[MainIndexEntry], position: int) -> MainIndexEntry | None:
    if position <= 0:
        return None
    try:
        return main_index[position - 1]
    except IndexError:
        return None


def _main_index_by_section(main_index: Iterable[MainIndexEntry], section_idx: int) -> list[MainIndexEntry]:
    return [entry for entry in main_index if entry.sectionIdx == section_idx]


# ---------------------------------------------------------------------------
# Info GUIDs, dynamic packets
# ---------------------------------------------------------------------------


def _read_info_guids(
    handle: BinaryIO,
    static_packets: list[StaticPacket],
    main_index: list[MainIndexEntry],
) -> list[dict[str, str]]:
    packet = _lookup_static(static_packets, idstr="InfoGuids")
    if packet is None:
        return []
    index_entries = _main_index_by_section(main_index, packet.index)
    if not index_entries:
        return []
    index_entry = index_entries[0]
    handle.seek(index_entry.offset, 0)
    count = index_entry.sectionL // 16
    result: list[dict[str, str]] = []
    for _ in range(int(count)):
        compact, pretty = _mixed_endian_guid(_read_exact(handle, 16))
        result.append({"guid": compact, "guidAsStr": pretty})
    return result


def _read_dynamic_packets(
    handle: BinaryIO,
    static_packets: list[StaticPacket],
    main_index: list[MainIndexEntry],
) -> list[dict[str, object]]:
    packet = _lookup_static(static_packets, idstr="InfoChangeStream")
    if packet is None:
        return []
    index_entry = _main_index_by_position(main_index, packet.index)
    if index_entry is None:
        return []

    handle.seek(index_entry.offset, 0)
    count = index_entry.sectionL // 48
    dynamic_packets: list[dict[str, object]] = []
    for i in range(int(count)):
        guid_compact, guid_pretty = _mixed_endian_guid(_read_exact(handle, 16))
        base_days = _read_double(handle)
        frac_days = _read_double(handle)
        seconds = base_days * DAY_SECONDS + frac_days - DATETIME_MINUS_FACTOR
        timestamp = _safe_posix_datetime(seconds)
        date_fraction = _read_double(handle)
        internal_offset_start = _read_u64(handle)
        packet_size = _read_u64(handle)
        dynamic_packets.append(
            {
                "offset": index_entry.offset + (i + 1) * 48,
                "guid": guid_compact,
                "guidAsStr": guid_pretty,
                "date": timestamp,
                "datefrac": date_fraction,
                "internalOffsetStart": internal_offset_start,
                "packetSize": packet_size,
                "data": b"",
                "IDStr": STATIC_PACKET_ID_MAP.get(guid_pretty, "UNKNOWN"),
            }
        )

    for packet in dynamic_packets:
        size = int(packet["packetSize"])
        if size <= 0:
            continue
        static_packet = _lookup_static(static_packets, tag=packet["guidAsStr"])
        if static_packet is None:
            continue
        sections = _main_index_by_section(main_index, static_packet.index)
        if not sections:
            continue

        remaining = size
        cursor = int(packet["internalOffsetStart"])
        collected = bytearray()
        accumulated_offset = 0
        for section in sections:
            section_start = accumulated_offset
            section_end = section_start + section.sectionL
            if cursor >= section_end:
                accumulated_offset = section_end
                continue
            read_start = max(cursor, section_start)
            read_end = min(cursor + remaining, section_end)
            if read_end <= read_start:
                accumulated_offset = section_end
                continue
            file_pos = section.offset + (read_start - section_start)
            handle.seek(file_pos, 0)
            chunk = _read_exact(handle, read_end - read_start)
            collected.extend(chunk)
            remaining -= len(chunk)
            accumulated_offset = section_end
            if remaining <= 0:
                break
        packet["data"] = bytes(collected)
    return dynamic_packets


# ---------------------------------------------------------------------------
# TSInfo parsing
# ---------------------------------------------------------------------------

TS_LABEL_SIZE = 64
LABEL_SIZE = 32
TS_ENTRY_RESERVED = 56
TS_ENTRY_STRIDE = 552


def _parse_tsinfo_entries(buffer: bytes, count_offset: int, data_offset: int) -> list[TSEntry]:
    if len(buffer) < count_offset + 4:
        return []
    count = int.from_bytes(buffer[count_offset : count_offset + 4], "little")
    entries: list[TSEntry] = []
    offset = data_offset
    for _ in range(count):
        chunk = buffer[offset : offset + TS_ENTRY_STRIDE]
        if len(chunk) < TS_ENTRY_STRIDE:
            break
        inner = 0
        # Label (64 bytes UTF-16)
        label = _decode_utf16(chunk[inner : inner + TS_LABEL_SIZE * 2])
        inner += TS_LABEL_SIZE * 2
        # Active sensor (32 bytes UTF-16)
        active = _decode_utf16(chunk[inner : inner + LABEL_SIZE * 2])
        inner += LABEL_SIZE * 2
        # Ref sensor (8 bytes UTF-16)
        ref = _decode_utf16(chunk[inner : inner + 4 * 2])
        inner += 4 * 2
        # Reserved (56 bytes)
        inner += TS_ENTRY_RESERVED
        # Low cut (8 bytes double)
        low_cut = struct.unpack_from("<d", chunk, inner)[0]
        inner += 8
        # High cut (8 bytes double)
        high_cut = struct.unpack_from("<d", chunk, inner)[0]
        inner += 8
        # Sampling rate (8 bytes double)
        sampling_rate = struct.unpack_from("<d", chunk, inner)[0]
        inner += 8
        # Resolution (8 bytes double)
        resolution = struct.unpack_from("<d", chunk, inner)[0]
        inner += 8
        # Special mark (2 bytes uint16)
        special_mark = struct.unpack_from("<H", chunk, inner)[0]
        inner += 2
        # Notch (2 bytes uint16)
        notch = struct.unpack_from("<H", chunk, inner)[0]
        inner += 2
        # EEG offset (8 bytes double)
        eeg_offset = struct.unpack_from("<d", chunk, inner)[0]
        entries.append(
            TSEntry(
                label=label,
                activeSensor=active,
                refSensor=ref,
                lowcut=low_cut,
                hiCut=high_cut,
                samplingRate=sampling_rate,
                resolution=resolution,
                specialMark=str(special_mark),
                notch=bool(notch),
                eeg_offset=int(eeg_offset),
            )
        )
        offset += TS_ENTRY_STRIDE
    return entries


def _read_tsinfo(
    handle: BinaryIO,
    static_packets: list[StaticPacket],
    dynamic_packets: list[dict[str, object]],
    main_index: list[MainIndexEntry],
) -> list[TSEntry]:
    dynamic_ts = [pkt for pkt in dynamic_packets if pkt.get("IDStr") == "TSGUID" and pkt.get("data")]
    if dynamic_ts:
        return _parse_tsinfo_entries(dynamic_ts[0]["data"], count_offset=752, data_offset=760)

    static_ts = _lookup_static(static_packets, idstr="TSGUID")
    if static_ts is None:
        raise ValueError("No TSInfo packet present in static or dynamic sections")
    index_entries = _main_index_by_section(main_index, static_ts.index)
    if not index_entries:
        raise ValueError("TSInfo static packet missing main-index entry")
    index_entry = index_entries[0]
    handle.seek(index_entry.offset, 0)
    _read_exact(handle, 16)  # GUID
    packet_length = _read_u64(handle)
    buffer = _read_exact(handle, packet_length)
    return _parse_tsinfo_entries(buffer, count_offset=728, data_offset=736)


# ---------------------------------------------------------------------------
# Segments and events
# ---------------------------------------------------------------------------

def _ole_to_datetime(value: float) -> datetime:
    seconds = value * DAY_SECONDS - DATETIME_MINUS_FACTOR
    return _safe_posix_datetime(seconds)


def _read_segments(
    handle: BinaryIO,
    static_packets: list[StaticPacket],
    main_index: list[MainIndexEntry],
    ts_info: list[TSEntry],
) -> list[SegmentInfo]:
    segment_packet = _lookup_static(static_packets, idstr="SegmentStream")
    if segment_packet is None:
        return []
    index_entries = _main_index_by_section(main_index, segment_packet.index)
    if not index_entries:
        return []
    index_entry = index_entries[0]
    handle.seek(index_entry.offset, 0)
    count = index_entry.sectionL // 152
    segments: list[SegmentInfo] = []
    for _ in range(int(count)):
        date_ole = _read_double(handle)
        date = _ole_to_datetime(date_ole)
        handle.seek(8, 1)
        duration = _read_double(handle)
        handle.seek(128, 1)
        sampling = np.array([entry.samplingRate for entry in ts_info], dtype=float)
        scale = np.array([entry.resolution for entry in ts_info], dtype=float)
        if np.allclose(scale, 0.0):
            scale = np.ones_like(scale)
        sample_counts = np.rint(sampling * duration).astype(int)
        segments.append(
            SegmentInfo(
                dateOLE=date_ole,
                date=date,
                duration=duration,
                chName=[entry.label for entry in ts_info],
                refName=[entry.refSensor for entry in ts_info],
                samplingRate=sampling,
                scale=scale,
                sampleCount=sample_counts,
            )
        )
    return segments


_EVENT_PACKET_GUID = bytes.fromhex("80F699B7A472D31193D300500400C148")
_EVENT_GUID_LABELS = {
    "{A5A95646-A7F8-11CF-831A-0800091B5BDA}": "Seizure",
    "{A5A95612-A7F8-11CF-831A-0800091B5BDA}": "Annotation",
    "{08784382-C765-11D3-90CE-00104B6F4F70}": "Format change",
    "{6FF394DA-D1B8-46DA-B78F-866C67CF02AF}": "Photic",
    "{481DFC97-013C-4BC5-A203-871B0375A519}": "Posthyperventilation",
    "{725798BF-CD1C-4909-B793-6C7864C27AB7}": "Review progress",
    "{96315D79-5C24-4A65-B334-E31A95088D55}": "Exam start",
    "{A5A95608-A7F8-11CF-831A-0800091B5BDA}": "Hyperventilation",
    "{A5A95617-A7F8-11CF-831A-0800091B5BDA}": "Impedance",
}


def _read_events(
    handle: BinaryIO,
    static_packets: list[StaticPacket],
    main_index: list[MainIndexEntry],
) -> list[EventItem]:
    event_packet = _lookup_static(static_packets, tag="Events")
    if event_packet is None:
        return []
    index_entries = _main_index_by_section(main_index, event_packet.index)
    if not index_entries:
        return []

    events: list[EventItem] = []
    for entry in index_entries:
        handle.seek(entry.offset, 0)
        guid = _read_exact(handle, 16)
        packet_length = _read_u64(handle)
        if guid != _EVENT_PACKET_GUID:
            continue
        packet_start = handle.tell()
        while handle.tell() - packet_start < packet_length:
            marker_start = handle.tell()
            handle.seek(8, 1)
            date_ole = _read_double(handle)
            date_fraction = _read_double(handle)
            duration = _read_double(handle)
            handle.seek(48, 1)
            user_raw = _read_exact(handle, 12 * 2)
            user = user_raw.decode("utf-16le", errors="ignore").rstrip("\x00 ")
            text_len = _read_u64(handle)
            _guid_compact, guid_pretty = _mixed_endian_guid(_read_exact(handle, 16))
            handle.seek(16, 1)
            label = _read_exact(handle, 32 * 2).decode("utf-16le", errors="ignore").rstrip("\x00 ")
            annotation = None
            if guid_pretty == "{A5A95612-A7F8-11CF-831A-0800091B5BDA}":
                handle.seek(32, 1)
                annotation_raw = _read_exact(handle, int(text_len) * 2)
                annotation = annotation_raw.decode("utf-16le", errors="ignore").rstrip("\x00 ")
            label_text = _EVENT_GUID_LABELS.get(guid_pretty, "UNKNOWN")
            event_time = _ole_to_datetime(date_ole + date_fraction / DAY_SECONDS)
            events.append(
                EventItem(
                    dateOLE=date_ole,
                    dateFraction=date_fraction,
                    date=event_time,
                    duration=duration,
                    user=user,
                    GUID=guid_pretty,
                    label=label,
                    IDStr=label_text,
                    annotation=annotation,
                )
            )
            consumed = handle.tell() - marker_start
            to_skip = 240 - consumed
            if to_skip > 0:
                handle.seek(to_skip, 1)
    return events


# ---------------------------------------------------------------------------
# Public header synthesis
# ---------------------------------------------------------------------------


def _infer_reference(segments: list[SegmentInfo]) -> str | None:
    if not segments:
        return None
    ref_labels = [ref for ref in segments[0].refName if ref]
    if not ref_labels:
        return None
    if sorted(set(ref_labels)) == ["REF"]:
        return "common"
    return "unknown"


def _build_public_header(nrv_header: NervusHeader) -> dict[str, object]:
    if not nrv_header.Segments:
        return {
            "Fs": None,
            "nChans": 0,
            "label": [],
            "nSamples": 0,
            "nSamplesPre": 0,
            "nTrials": 0,
            "reference": nrv_header.reference,
            "filename": str(nrv_header.filename),
        }

    sampling_rates = nrv_header.Segments[0].samplingRate
    rounded = np.round(sampling_rates, decimals=6)
    unique_rates, counts = np.unique(rounded, return_counts=True)
    target_sampling_rate = float(unique_rates[np.argmax(counts)])
    match_mask = np.isclose(rounded, target_sampling_rate, rtol=1e-6, atol=1e-6)
    matching_channels = np.where(match_mask)[0]
    if matching_channels.size == 0:
        match_mask = np.ones_like(sampling_rates, dtype=bool)
        matching_channels = np.arange(len(sampling_rates))

    nrv_header.targetSamplingRate = target_sampling_rate
    nrv_header.matchingChannels = (matching_channels + 1).tolist()
    nrv_header.excludedChannels = (np.where(~match_mask)[0] + 1).tolist()
    nrv_header.targetNumberOfChannels = len(matching_channels)

    total_samples = 0
    for segment in nrv_header.Segments:
        if segment.sampleCount.size == 0:
            continue
        total_samples += int(np.max(segment.sampleCount[match_mask]))
    nrv_header.targetSampleCount = total_samples

    labels = [nrv_header.Segments[0].chName[idx] for idx in matching_channels]
    nrv_header.reference = _infer_reference(nrv_header.Segments)
    nrv_header.startDateTime = nrv_header.Segments[0].date if nrv_header.Segments else None

    return {
        "Fs": target_sampling_rate,
        "nChans": len(labels),
        "label": labels,
        "nSamples": total_samples,
        "nSamplesPre": 0,
        "nTrials": 1,
        "reference": nrv_header.reference,
        "filename": str(nrv_header.filename),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_nervus_header(path: str | Path):
    filename = Path(path)
    with filename.open("rb") as handle:
        _ = [_read_u32(handle) for _ in range(5)]
        _read_u32(handle)
        index_idx = _read_u32(handle)
        if index_idx == 0:
            raise ValueError("Unsupported old-style Nicolet file format (pre-ca. 2012)")

        static_packets = _read_static_packets(handle)
        qi_index = _read_qi_index(handle, len(static_packets))
        qi_index2 = _read_qi_index2(handle, qi_index)
        main_index = _read_main_index(handle, index_idx, int(qi_index["nrEntries"]))
        info_guids = _read_info_guids(handle, static_packets, main_index)
        dynamic_packets = _read_dynamic_packets(handle, static_packets, main_index)
        ts_info = _read_tsinfo(handle, static_packets, dynamic_packets, main_index)
        segments = _read_segments(handle, static_packets, main_index, ts_info)
        events = _read_events(handle, static_packets, main_index)

    nrv_header = NervusHeader(
        filename=filename,
        StaticPackets=static_packets,
        QIIndex=qi_index,
        QIIndex2=qi_index2,
        MainIndex=main_index,
        allIndexIDs=[entry.sectionIdx for entry in main_index],
        infoGuids=info_guids,
        DynamicPackets=dynamic_packets,
        TSInfo=ts_info,
        Segments=segments,
        Events=events,
    )

    public_header = _build_public_header(nrv_header)
    return public_header, nrv_header
