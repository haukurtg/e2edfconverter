"""Header parsing utilities for Nicolet `.e` (and legacy `.eeg`) recordings."""

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

# Patient info property order (matches FieldTrip/Nicolet Reader)
INFO_PROPS = [
    "patientID",
    "firstName",
    "middleName",
    "lastName",
    "altID",
    "mothersMaidenName",
    "DOB",
    "DOD",
    "street",
    "sexID",
    "phone",
    "notes",
    "dominance",
    "siteID",
    "suffix",
    "prefix",
    "degree",
    "apartment",
    "city",
    "state",
    "country",
    "language",
    "height",
    "weight",
    "race",
    "religion",
    "maritalStatus",
]


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
    "{8E94EF21-70F5-11D3-8F72-00105A9AFD56}": "FILEINFOGUID",
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


def _read_patient_info(
    handle: BinaryIO,
    static_packets: list[StaticPacket],
    main_index: list[MainIndexEntry],
) -> dict[str, object] | None:
    packet = _lookup_static(static_packets, idstr="PATIENTINFOGUID")
    if packet is None:
        return None
    index_entries = _main_index_by_section(main_index, packet.index)
    if not index_entries:
        return None
    index_entry = index_entries[0]
    handle.seek(index_entry.offset, 0)
    _read_exact(handle, 16)  # GUID
    _read_u64(handle)  # section length
    n_values = _read_u64(handle)
    n_bstr = _read_u64(handle)

    info: dict[str, object] = {}
    for _ in range(int(n_values)):
        prop_id = _read_u64(handle)
        if prop_id in (7, 8, 23, 24):
            value = _read_double(handle)
            if prop_id in (7, 8):
                # Convert OLE days to datetime
                info[INFO_PROPS[prop_id - 1]] = _ole_to_datetime(value)
            else:
                info[INFO_PROPS[prop_id - 1]] = value
        else:
            # Non-numeric properties are stored in the BSTR block
            if 1 <= prop_id <= len(INFO_PROPS):
                info[INFO_PROPS[prop_id - 1]] = None

    if n_bstr:
        str_setup = [_read_u64(handle) for _ in range(int(n_bstr) * 2)]
        for i in range(0, len(str_setup), 2):
            prop_id = int(str_setup[i])
            strlen = int(str_setup[i + 1])
            raw = _read_exact(handle, (strlen + 1) * 2)
            value = _decode_utf16(raw)
            if 1 <= prop_id <= len(INFO_PROPS):
                info[INFO_PROPS[prop_id - 1]] = value

    for prop in INFO_PROPS:
        info.setdefault(prop, None)
    return info


def _read_signal_info(
    handle: BinaryIO,
    static_packets: list[StaticPacket],
    main_index: list[MainIndexEntry],
) -> list[dict[str, object]]:
    packet = _lookup_static(static_packets, idstr="SIGNALINFOGUID")
    if packet is None:
        return []
    index_entries = _main_index_by_section(main_index, packet.index)
    if not index_entries:
        return []
    index_entry = index_entries[0]
    handle.seek(index_entry.offset, 0)
    _read_exact(handle, 16)  # GUID
    _read_exact(handle, 64)  # name
    handle.seek(152, 1)
    handle.seek(512, 1)
    nr_idx = _read_u16(handle)
    _read_exact(handle, 3 * 2)  # misc1

    signals: list[dict[str, object]] = []
    for _ in range(int(nr_idx)):
        sensor_name = _read_utf16(handle, 32)
        transducer = _read_utf16(handle, 16)
        guid_raw = _read_exact(handle, 16)
        guid_compact, guid_pretty = _mixed_endian_guid(guid_raw)
        b_bipolar = _read_u32(handle)
        b_ac = _read_u32(handle)
        b_high_filter = _read_u32(handle)
        color = _read_u32(handle)
        _read_exact(handle, 256)
        signals.append(
            {
                "sensorName": sensor_name,
                "transducer": transducer,
                "guid": guid_compact,
                "guidAsStr": guid_pretty,
                "bBiPolar": bool(b_bipolar),
                "bAC": bool(b_ac),
                "bHighFilter": bool(b_high_filter),
                "color": color,
            }
        )
    return signals


def _read_channel_info(
    handle: BinaryIO,
    static_packets: list[StaticPacket],
    main_index: list[MainIndexEntry],
) -> list[dict[str, object]]:
    packet = _lookup_static(static_packets, idstr="CHANNELGUID")
    if packet is None:
        return []
    index_entries = _main_index_by_section(main_index, packet.index)
    if not index_entries:
        return []
    index_entry = index_entries[0]
    handle.seek(index_entry.offset, 0)
    _read_exact(handle, 16)  # GUID
    _read_exact(handle, 64)  # name
    handle.seek(152, 1)
    _read_exact(handle, 16)  # reserved
    _read_exact(handle, 16)  # deviceID
    handle.seek(488, 1)
    nr_idx1 = _read_u32(handle)
    nr_idx2 = _read_u32(handle)

    channel_info: list[dict[str, object]] = []
    current_index = 0
    for _ in range(int(nr_idx2)):
        sensor = _read_utf16(handle, 32)
        sampling_rate = _read_double(handle)
        b_on = _read_u32(handle)
        l_input_id = _read_u32(handle)
        l_input_setting_id = _read_u32(handle)
        _read_exact(handle, 4)  # reserved
        handle.seek(128, 1)
        if b_on:
            index_id = current_index
            current_index += 1
        else:
            index_id = -1
        channel_info.append(
            {
                "sensor": sensor,
                "samplingRate": sampling_rate,
                "bOn": bool(b_on),
                "lInputID": l_input_id,
                "lInputSettingID": l_input_setting_id,
                "indexID": index_id,
            }
        )
    # nr_idx1 is often the number of traces; keep it for diagnostics.
    if nr_idx1 and len(channel_info) == 0:
        handle.seek(index_entry.offset + 16 + 64 + 152 + 16 + 16 + 488, 0)
    return channel_info


def _read_montage_info(
    handle: BinaryIO,
    static_packets: list[StaticPacket],
    main_index: list[MainIndexEntry],
) -> list[dict[str, object]]:
    montage_packet = _lookup_static(static_packets, idstr="DERIVATIONGUID")
    if montage_packet is None:
        return []
    index_entries = _main_index_by_section(main_index, montage_packet.index)
    if not index_entries:
        return []
    index_entry = index_entries[0]
    handle.seek(index_entry.offset + 40, 0)
    montage_name = _read_utf16(handle, 32)
    handle.seek(640, 1)
    num_derivations = _read_u32(handle)
    _read_u32(handle)
    montage: list[dict[str, object]] = []
    for _ in range(int(num_derivations)):
        derivation_name = _read_utf16(handle, 64)
        signal_name_1 = _read_utf16(handle, 32)
        signal_name_2 = _read_utf16(handle, 32)
        handle.seek(264, 1)
        montage.append(
            {
                "montageName": montage_name,
                "derivationName": derivation_name,
                "signalName1": signal_name_1,
                "signalName2": signal_name_2,
            }
        )

    # Attach display colors when available.
    display_packet = _lookup_static(static_packets, idstr="DISPLAYGUID")
    if display_packet:
        display_entries = _main_index_by_section(main_index, display_packet.index)
        if display_entries:
            display_entry = display_entries[0]
            handle.seek(display_entry.offset + 40, 0)
            display_name = _read_utf16(handle, 32)
            handle.seek(640, 1)
            num_traces = _read_u32(handle)
            _read_u32(handle)
            if num_traces == num_derivations:
                for i in range(int(num_traces)):
                    handle.seek(32, 1)
                    color = _read_u32(handle)
                    handle.seek(132, 1)
                    montage[i]["displayName"] = display_name
                    montage[i]["color"] = color
    return montage


def _read_dynamic_montages(dynamic_packets: list[dict[str, object]]) -> list[dict[str, object]]:
    montages: list[dict[str, object]] = []
    montage_packets = [pkt for pkt in dynamic_packets if pkt.get("IDStr") == "DERIVATIONGUID" and pkt.get("data")]
    for pkt in montage_packets:
        data = pkt.get("data") or b""
        if len(data) < 760:
            continue
        guid1 = _mixed_endian_guid(data[0:16])[0]
        packet_size = int.from_bytes(data[16:24], "little")
        guid2 = _mixed_endian_guid(data[24:40])[0]
        item_name = _decode_utf16(data[40:104])
        elements = int.from_bytes(data[744:748], "little")
        offset = 752
        channels = []
        for _ in range(elements):
            name = _decode_utf16(data[offset : offset + 128])
            offset += 128
            active = _decode_utf16(data[offset : offset + 64])
            offset += 64
            reference = _decode_utf16(data[offset : offset + 64])
            offset += 64
            is_derived = bool(data[offset])
            offset += 1
            is_special = bool(data[offset])
            offset += 1
            offset += 256 + 6
            channels.append(
                {
                    "name": name,
                    "active": active,
                    "reference": reference,
                    "isDerived": is_derived,
                    "isSpecial": is_special,
                }
            )
        montages.append(
            {
                "guid1": guid1,
                "guid2": guid2,
                "packetSize": packet_size,
                "itemName": item_name,
                "channels": channels,
            }
        )
    return montages


# ---------------------------------------------------------------------------
# TSInfo parsing
# ---------------------------------------------------------------------------

TS_LABEL_SIZE = 64
LABEL_SIZE = 32
TS_ENTRY_RESERVED = 56
TS_ENTRY_STRIDE = 552


def _ts_entries_sane(entries: list[TSEntry]) -> bool:
    if not entries:
        return False
    sampling = np.array([entry.samplingRate for entry in entries], dtype=float)
    resolution = np.array([entry.resolution for entry in entries], dtype=float)
    valid_sampling = np.isfinite(sampling) & (sampling > 0.1) & (sampling < 200_000)
    valid_resolution = np.isfinite(resolution) & (np.abs(resolution) < 1e9)
    valid = valid_sampling & valid_resolution
    return int(np.sum(valid)) >= max(1, len(entries) // 2)


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
        if not np.isfinite(eeg_offset) or abs(eeg_offset) > 1e9:
            # Some legacy layouts store EEG offset as float32 with padding.
            eeg_offset_f32 = struct.unpack_from("<f", chunk, inner)[0]
            if np.isfinite(eeg_offset_f32) and abs(eeg_offset_f32) <= 1e6:
                eeg_offset = float(eeg_offset_f32)
            else:
                eeg_offset = 0.0
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
                eeg_offset=float(eeg_offset),
            )
        )
        offset += TS_ENTRY_STRIDE
    return entries


def _read_tsinfo_packets(
    handle: BinaryIO,
    static_packets: list[StaticPacket],
    dynamic_packets: list[dict[str, object]],
    main_index: list[MainIndexEntry],
) -> list[dict[str, object]]:
    packets: list[dict[str, object]] = []

    dynamic_ts = [pkt for pkt in dynamic_packets if pkt.get("IDStr") == "TSGUID" and pkt.get("data")]
    for pkt in dynamic_ts:
        entries = _parse_tsinfo_entries(pkt["data"], count_offset=752, data_offset=760)
        if _ts_entries_sane(entries):
            packets.append({"date": pkt.get("date"), "entries": entries, "source": "dynamic"})

    if packets:
        return packets

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
    entries = _parse_tsinfo_entries(buffer, count_offset=728, data_offset=736)
    packets.append({"date": None, "entries": entries, "source": "static"})
    return packets


# ---------------------------------------------------------------------------
# Segments and events
# ---------------------------------------------------------------------------

def _ole_to_datetime(value: float) -> datetime:
    seconds = value * DAY_SECONDS - DATETIME_MINUS_FACTOR
    return _safe_posix_datetime(seconds)


def _select_tsinfo_for_segment(
    segment_date: datetime,
    ts_packets: list[dict[str, object]],
) -> list[TSEntry]:
    if not ts_packets:
        return []
    if len(ts_packets) == 1:
        return ts_packets[0]["entries"]

    dated_packets = [pkt for pkt in ts_packets if pkt.get("date")]
    if not dated_packets:
        return ts_packets[0]["entries"]

    # Choose the most recent TSInfo packet at or before the segment start.
    dated_packets.sort(key=lambda pkt: pkt["date"])
    selected = dated_packets[0]
    for pkt in dated_packets:
        if pkt["date"] <= segment_date:
            selected = pkt
        else:
            break
    return selected["entries"]


def _read_segments(
    handle: BinaryIO,
    static_packets: list[StaticPacket],
    main_index: list[MainIndexEntry],
    ts_packets: list[dict[str, object]],
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

        ts_info = _select_tsinfo_for_segment(date, ts_packets)
        sampling = np.array([entry.samplingRate for entry in ts_info], dtype=float)
        scale = np.array([entry.resolution for entry in ts_info], dtype=float)
        eeg_offset = np.array([entry.eeg_offset for entry in ts_info], dtype=float)
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
                eegOffset=eeg_offset,
            )
        )
    return segments


_EVENT_PACKET_GUID = bytes.fromhex("80F699B7A472D31193D300500400C148")
_EVENT_GUID_LABELS = {
    "{A5A95646-A7F8-11CF-831A-0800091B5BDA}": "Seizure",
    "{A5A95612-A7F8-11CF-831A-0800091B5BDA}": "Annotation",
    "{A5A95645-A7F8-11CF-831A-0800091B5BDA}": "Event Comment",
    "{A5A95600-A7F8-11CF-831A-0800091B5BDA}": "Øyne åpnes",
    "{A5A95601-A7F8-11CF-831A-0800091B5BDA}": "Øyne lukkes",
    "{A5A95602-A7F8-11CF-831A-0800091B5BDA}": "Bevegelse",
    "{A5A95605-A7F8-11CF-831A-0800091B5BDA}": "Snakker",
    "{A5A95606-A7F8-11CF-831A-0800091B5BDA}": "Døsig",
    "{A5A95608-A7F8-11CF-831A-0800091B5BDA}": "Hyperventilering",
    "{A5A95617-A7F8-11CF-831A-0800091B5BDA}": "Impedanse",
    "{739802CE-E11A-4627-A847-CB72BDBFAF2D}": "Tiltale",
    "{08784382-C765-11D3-90CE-00104B6F4F70}": "Format change",
    "{6FF394DA-D1B8-46DA-B78F-866C67CF02AF}": "Photic",
    "{481DFC97-013C-4BC5-A203-871B0375A519}": "Post Hyperventilering",
    "{725798BF-CD1C-4909-B793-6C7864C27AB7}": "Review progress",
    "{96315D79-5C24-4A65-B334-E31A95088D55}": "Us. start",
    "{98FB933E-5183-4E4D-99AF-88AA29B22D05}": "Detections Active",
    "{59508CBF-8E0A-43BC-9D80-49E25B14395C}": "Utbrudd",
    "{0FF74532-1AE0-4970-A484-5A89C1307A0E}": "EEG endring",
    "{3EF5BC5D-E6EA-4933-B991-38FF6F3E881A}": "SW",
    "{EFAFE7DC-C170-4D16-91BA-7582BD45A47B}": "Se oppover",
    "{2AACF05C-2FD4-4AE9-89E7-0586CEF59280}": "Se nedover",
    "{5AFD92A5-1079-4758-8916-B5CCDD1AF49B}": "Se til Høyre",
    "{28871ADA-BDDC-4119-B2A6-487468579E74}": "Se til Venstre",
    "{5EFEE8AB-5172-4837-BFA1-514AA764F0E8}": "Gaper",
    "{32F2469E-6792-4CAD-8E11-B7747688BB8B}": "Video Start",
    "{056F522F-DDA5-48B9-82E1-1A75C35CBC30}": "Video Stop",
    "{A5A95625-A7F8-11CF-831A-0800091B5BDA}": "Lys av",
    "{A5A95626-A7F8-11CF-831A-0800091B5BDA}": "Lys på",
    "{A71A6DB5-4150-48BF-B462-1C40521EBD6F}": "Forsterker frakoblet",
    "{6387C7C8-6F98-4886-9AF4-FA750ED300DE}": "Amplifier Reconnected",
    "{71EECE80-EBC4-41C7-BF26-E56911426FB4}": "Recording Paused",
    "{C3B68051-EDCF-418C-8D53-27077B92DE22}": "Spike",
    "{99FFE0AA-B8F9-49E5-8390-8F072F4E00FC}": "EEG Check",
    "{A5A9560A-A7F8-11CF-831A-0800091B5BDA}": "Print",
    "{A5A95616-A7F8-11CF-831A-0800091B5BDA}": "Patient Event",
    "{28C9D814-5B90-4BDD-A70C-9BC27D1C35A7}": "Funn: langsom aktivitet",
    "{0EECCD4A-D95D-46A0-BFCB-22F690778BA4}": "Funn",
    "{0DE05C94-7D03-47B9-864F-D586627EA891}": "Eyes closed",
    "{583AA2C6-1F4E-47CF-A8D4-80C69EB8A5F3}": "Eyes open",
    "{BAE4550A-8409-4289-9D8A-0D571A206BEC}": "Eating",
    "{1F3A45A4-4D0F-4CC4-A43A-CAD2BC2D71F2}": "ECG",
    "{B0BECF64-E669-42B1-AE20-97A8B0BBEE26}": "Toilet",
    "{A5A95611-A7F8-11CF-831A-0800091B5BDA}": "Fix Electrode",
    "{08EC3F49-978D-4FE4-AE77-4C421335E5FF}": "Prune",
    "{0A205CD4-1480-4F02-8AFF-4E4CD3B21078}": "Artifact",
    "{A5A95609-A7F8-11CF-831A-0800091B5BDA}": "Print D",
    "{A5A95637-A7F8-11CF-831A-0800091B5BDA}": "Tachycardia",
    "{A0172995-4A24-401C-AB68-B585474E4C07}": "Seizure",
    "{FF37D596-5703-43F9-A3F3-FA572C5D958C}": "Spike wave",
    "{9DF82C59-6520-46E5-940F-16B1282F3DD6}": "EEG Check-theta li T",
    "{06519E79-3C7B-4535-BA76-2AD76B6C65C8}": "Kom.-*",
    "{CA4FCAD4-802E-4214-881A-E9C1C6549ABD}": "Arousal",
    "{A5A95603-A7F8-11CF-831A-0800091B5BDA}": "Blunker",
    "{77A38C02-DCD4-4774-A47D-40437725B278}": "+Anfallsmuster D-?",
    "{32DB96B9-ED12-429A-B98D-27B2A82AD61F}": "spike wave",
    "{24387A0E-AA04-40B4-82D4-6D58F24D59AB}": "Anfallsmuster",
    "{A5A95636-A7F8-11CF-831A-0800091B5BDA}": "Bradycardia",
    "{93A2CB2C-F420-4672-AA62-18989F768519}": "Detections Inactive",
    "{8C5D49BA-7105-4355-BF6C-B35B9A4E594A}": "EEG-Check",
    "{5A946B85-2E1D-46B8-9FB2-C0519C9BE681}": "Zaehneputzen",
    "{48DA028A-5264-4620-AD03-C8787951E237}": "Bewegt",
    "{C15CFF61-0326-4276-A08F-0BFC2354E7CC}": "Kratzt",
    "{F4DD5874-23BA-4FFA-94DD-BE436BB6910F}": "Anfall",
    "{A5A95610-A7F8-11CF-831A-0800091B5BDA}": "Flash",
    "{8CB92AA7-A886-4013-8D52-6CD1C71C72B4}": "ETP",
}


def _read_events(
    handle: BinaryIO,
    static_packets: list[StaticPacket],
    main_index: list[MainIndexEntry],
    segments: Sequence[SegmentInfo] | None = None,
) -> list[EventItem]:
    """Read events from the Events section.
    
    Each event is stored in its own packet with structure:
    - 16 bytes: packet GUID (must match _EVENT_PACKET_GUID)
    - 8 bytes: packet length
    - 240 bytes: event marker data
    
    Events are read sequentially until a non-matching GUID is found.
    """
    event_packet = _lookup_static(static_packets, tag="Events")
    if event_packet is None:
        return []
    index_entries = _main_index_by_section(main_index, event_packet.index)
    if not index_entries:
        return []

    events: list[EventItem] = []
    
    # Start at the first event section
    for entry_index, entry in enumerate(index_entries):
        offset = entry.offset
        if entry_index == 1:
            offset += 248
        section_end = entry.offset + entry.sectionL
        
        # Read packets sequentially until we hit a non-event GUID or section end
        while offset < section_end:
            handle.seek(offset, 0)
            
            # Read packet header: GUID (16 bytes) + length (8 bytes)
            guid = _read_exact(handle, 16)
            packet_length = _read_u64(handle)
            
            # Stop if this is not an event packet
            if guid != _EVENT_PACKET_GUID:
                break
            if packet_length <= 0 or packet_length > section_end - offset or packet_length > 1_000_000:
                break
            if packet_length < 240:
                break
            
            # Read ONE event from this packet
            # Skip eventID (8 bytes, not used)
            handle.seek(8, 1)
            
            date_ole = _read_double(handle)
            date_fraction = _read_double(handle)
            duration = _read_double(handle)
            
            # Skip 48 bytes reserved
            handle.seek(48, 1)
            
            # User (12 UTF-16 chars = 24 bytes)
            user_raw = _read_exact(handle, 12 * 2)
            user = user_raw.decode("utf-16le", errors="ignore").rstrip("\x00 ")
            
            # Text length for annotations
            text_len = _read_u64(handle)
            
            # Event type GUID
            _guid_compact, guid_pretty = _mixed_endian_guid(_read_exact(handle, 16))
            
            # Skip Reserved4 (16 bytes)
            handle.seek(16, 1)
            
            # Label (32 UTF-16 chars = 64 bytes)
            label = _read_exact(handle, 32 * 2).decode("utf-16le", errors="ignore").rstrip("\x00 ")
            if label == "-":
                label = ""
            
            # Read optional text payload (used by annotations and some system events).
            annotation = None
            if text_len > 0:
                handle.seek(32, 1)
                bytes_left = offset + packet_length - handle.tell()
                max_chars = max(int(bytes_left // 2), 0)
                read_chars = min(int(text_len), max_chars)
                annotation_raw = _read_exact(handle, read_chars * 2) if read_chars else b""
                decoded = annotation_raw.decode("utf-16le", errors="ignore") if annotation_raw else ""
                if "\x00" in decoded:
                    decoded = decoded.split("\x00")[0]
                annotation = decoded.strip() or None
            
            label_text = _EVENT_GUID_LABELS.get(guid_pretty, "UNKNOWN")
            # Prefer richer photic labels when frequency text is present.
            if label_text == "Photic" and annotation:
                label = f"{label_text} - {annotation}"
                annotation = None
            if label_text == "Format change" and annotation:
                label_text = "Forandring !"
                label = f"{label_text} - {annotation}"
                annotation = None
            if label_text == "Recording Paused" and annotation:
                label = f"{label_text} - {annotation}"
                annotation = None
            if label_text == "Event Comment" and annotation and not label:
                label = annotation
                annotation = None
            event_time = _ole_to_datetime(date_ole + date_fraction / DAY_SECONDS)
            seg_index = None
            if segments:
                seg_index = 0
                for idx, segment in enumerate(segments):
                    if segment.date <= event_time:
                        seg_index = idx
                    else:
                        break
            is_epoch = duration > 0.0
            
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
                    segmentIndex=seg_index,
                    isEpoch=is_epoch,
                )
            )
            
            # Move to next packet (offset + packet_length)
            offset = offset + packet_length
    
    return events


def _guid_to_mixed_bytes(guid_pretty: str) -> bytes:
    compact = guid_pretty.strip("{}").replace("-", "")
    raw = bytes.fromhex(compact)
    part1 = raw[0:4][::-1]
    part2 = raw[4:6][::-1]
    part3 = raw[6:8][::-1]
    remainder = raw[8:]
    return part1 + part2 + part3 + remainder


def _looks_like_label(text: str) -> bool:
    if not text or len(text) < 3:
        return False
    letters = sum(ch.isalpha() for ch in text)
    printable = sum(32 <= ord(ch) <= 126 or ch.isalpha() for ch in text)
    return letters >= 3 and printable / max(len(text), 1) > 0.6


def _scan_utf16_label(buffer: bytes, start: int, max_scan: int = 512, max_chars: int = 64) -> str | None:
    end_limit = min(len(buffer), start + max_scan)
    for offset in range(start, end_limit, 2):
        end = offset
        chars = 0
        while end + 1 < len(buffer) and chars < max_chars:
            if buffer[end : end + 2] == b"\x00\x00":
                break
            end += 2
            chars += 1
        if end == offset:
            continue
        text = _decode_utf16(buffer[offset:end])
        if _looks_like_label(text):
            return text
    return None


def _clean_event_label(text: str) -> str:
    if not text:
        return ""
    cleaned = "".join(ch if ch.isascii() and ch.isprintable() else " " for ch in text)
    cleaned = " ".join(cleaned.split())
    if cleaned in {"yne pne", "yne åpne"}:
        return "Øyne åpnes"
    if cleaned in {"yne lukket", "yne lukkes"}:
        return "Øyne lukkes"
    if cleaned == "Format forandring.":
        return "Forandring !"
    if cleaned.startswith("Impedance"):
        return "Impedanse"
    if cleaned.startswith("Impedanse"):
        return "Impedanse"
    if cleaned == "Lys p":
        return "Lys på"
    if cleaned.startswith("Impedance"):
        return "Impedance"
    if cleaned.startswith("Automatic Detections Active"):
        return "Detections Active"
    if cleaned == "Video Recording Start":
        return "Video Start"
    if cleaned == "Video Recording Stop":
        return "Video Stop"
    if cleaned.startswith("Anfall "):
        return "Anfall"
    if cleaned == "Sleep Spindle":
        return "Spindle"
    return cleaned


def _looks_like_channel_label(text: str) -> bool:
    if not text:
        return False
    candidate = text.strip()
    if len(candidate) > 16 or any(ch.isspace() for ch in candidate):
        return False
    upper = candidate.upper()
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-+")
    if any(ch not in allowed for ch in upper):
        return False
    if not any(ch.isdigit() for ch in upper):
        return False
    return True


def _read_event_type_info(
    handle: BinaryIO,
    static_packets: list[StaticPacket],
    main_index: list[MainIndexEntry],
    target_guids: Iterable[str] | None = None,
) -> dict[str, str]:
    packet = _lookup_static(static_packets, idstr="EVENTTYPEINFOGUID")
    if packet is None:
        return {}
    index_entries = _main_index_by_section(main_index, packet.index)
    if not index_entries:
        return {}
    index_entry = index_entries[0]
    handle.seek(index_entry.offset, 0)
    buffer = _read_exact(handle, index_entry.sectionL)
    if not target_guids:
        return {}

    mapping: dict[str, str] = {}
    for guid in target_guids:
        mixed = _guid_to_mixed_bytes(guid)
        pos = buffer.find(mixed)
        while pos != -1:
            label = _scan_utf16_label(buffer, pos + 16)
            if label:
                cleaned = _clean_event_label(label)
                mapping[guid] = cleaned or label
                break
            pos = buffer.find(mixed, pos + 1)
    return mapping


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
    on_mask = np.ones_like(sampling_rates, dtype=bool)
    if nrv_header.ChannelInfo and len(nrv_header.ChannelInfo) == len(sampling_rates):
        on_mask = np.array([bool(ch.get("bOn", True)) for ch in nrv_header.ChannelInfo], dtype=bool)
    rounded = np.round(sampling_rates, decimals=6)
    unique_rates, counts = np.unique(rounded, return_counts=True)
    target_sampling_rate = float(unique_rates[np.argmax(counts)])
    match_mask = np.isclose(rounded, target_sampling_rate, rtol=1e-6, atol=1e-6) & on_mask
    matching_channels = np.where(match_mask)[0]
    if matching_channels.size == 0:
        match_mask = on_mask if np.any(on_mask) else np.ones_like(sampling_rates, dtype=bool)
        matching_channels = np.arange(len(sampling_rates))

    nrv_header.targetSamplingRate = target_sampling_rate
    nrv_header.matchingChannels = (matching_channels + 1).tolist()
    nrv_header.excludedChannels = (np.where(~match_mask)[0] + 1).tolist()
    nrv_header.targetNumberOfChannels = len(matching_channels)

    total_samples = 0
    for segment in nrv_header.Segments:
        if segment.sampleCount.size == 0:
            continue
        try:
            total_samples += int(np.max(segment.sampleCount[match_mask]))
        except ValueError:
            total_samples += int(np.max(segment.sampleCount))
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
    if filename.suffix.lower() == ".eeg":
        from .legacy_eeg import read_legacy_header as read_legacy_eeg_header

        try:
            legacy_header = read_legacy_eeg_header(filename)
        except Exception as exc:
            raise ValueError(
                "Unsupported legacy Nicolet file format (pre-ca. 2012)"
            ) from exc
        public_header = _build_public_header(legacy_header)
        return public_header, legacy_header
    with filename.open("rb") as handle:
        _ = [_read_u32(handle) for _ in range(5)]
        _read_u32(handle)
        index_idx = _read_u32(handle)
        if index_idx == 0:
            # Some pre-2012 .e files keep a zero header index, but still carry
            # modern static/QI tables at fixed offsets. Try modern parsing first.
            try:
                file_size = filename.stat().st_size
            except OSError:
                file_size = 0
            try:
                static_packets = _read_static_packets(handle)
                qi_index = _read_qi_index(handle, len(static_packets))
                qi_index_idx = int(qi_index.get("indexIdx", 0) or 0)
                nr_entries = int(qi_index.get("nrEntries", 0) or 0)
                if (
                    qi_index_idx > 0
                    and nr_entries > 0
                    and (file_size == 0 or qi_index_idx < file_size)
                ):
                    main_index = _read_main_index(handle, qi_index_idx, nr_entries)
                    info_guids = _read_info_guids(handle, static_packets, main_index)
                    dynamic_packets = _read_dynamic_packets(handle, static_packets, main_index)
                    patient_info = _read_patient_info(handle, static_packets, main_index)
                    sig_info = _read_signal_info(handle, static_packets, main_index)
                    channel_info = _read_channel_info(handle, static_packets, main_index)
                    montage_info = _read_montage_info(handle, static_packets, main_index)
                    montage_info2 = _read_dynamic_montages(dynamic_packets)
                    ts_packets = _read_tsinfo_packets(
                        handle, static_packets, dynamic_packets, main_index
                    )
                    segments = _read_segments(handle, static_packets, main_index, ts_packets)
                    events = _read_events(handle, static_packets, main_index, segments)
                    event_type_info = _read_event_type_info(
                        handle,
                        static_packets,
                        main_index,
                        target_guids={event.GUID for event in events},
                    )
                    if event_type_info:
                        for event in events:
                            if event.GUID in event_type_info and event.IDStr == "UNKNOWN":
                                event.IDStr = event_type_info[event.GUID]
                    for event in events:
                        if event.label and not any(ch.isascii() and ch.isalnum() for ch in event.label):
                            event.label = None
                        if event.IDStr == "UNKNOWN":
                            cleaned = _clean_event_label(event.label or "")
                            if cleaned and cleaned != "-":
                                event.IDStr = cleaned
                            else:
                                raw = event.label.strip() if event.label else ""
                                if raw and raw != "-":
                                    event.IDStr = raw
                        if event.IDStr in {"Detections Active", "Detections Inactive"} and event.annotation:
                            event.label = f"{event.IDStr} - {event.annotation}"
                            event.annotation = None
                        if event.IDStr == "Seizure" and (not event.label or "Anfall" in event.label):
                            event.IDStr = "Anfall"
                        if event.IDStr == "Funn" and event.label and _looks_like_channel_label(event.label):
                            event.label = None
                        if event.IDStr in {"UTBRUDD", "Utbrudd"}:
                            event.IDStr = "Utbrudd"
                            if event.label and _looks_like_channel_label(event.label):
                                event.label = None
                        if event.IDStr == "SW" and event.label and _looks_like_channel_label(event.label):
                            event.label = None
                        if event.IDStr == "Prune" and event.label:
                            lowered = event.label.lower()
                            if _looks_like_channel_label(event.label) or "marks epochs" in lowered:
                                event.label = None
                    header = NervusHeader(
                        filename=filename,
                        StaticPackets=static_packets,
                        QIIndex=qi_index,
                        QIIndex2=[],
                        MainIndex=main_index,
                        allIndexIDs=[entry.sectionIdx for entry in main_index],
                        infoGuids=info_guids,
                        DynamicPackets=dynamic_packets,
                        PatientInfo=patient_info,
                        SigInfo=sig_info,
                        ChannelInfo=channel_info,
                        TSInfo=ts_packets[0]["entries"] if ts_packets else [],
                        TSInfoBySegment=[_select_tsinfo_for_segment(seg.date, ts_packets) for seg in segments],
                        Segments=segments,
                        Events=events,
                        MontageInfo=montage_info,
                        MontageInfo2=montage_info2,
                        EventTypeInfo=event_type_info,
                        format="nicolet-e",
                    )
                    public_header = _build_public_header(header)
                    return public_header, header
            except Exception:
                pass
            try:
                from .legacy import read_legacy_header

                legacy_header = read_legacy_header(filename)
            except Exception as exc:
                raise ValueError(
                    "Unsupported legacy Nicolet file format (pre-ca. 2012)"
                ) from exc
            public_header = _build_public_header(legacy_header)
            return public_header, legacy_header

        static_packets = _read_static_packets(handle)
        qi_index = _read_qi_index(handle, len(static_packets))
        qi_index2 = _read_qi_index2(handle, qi_index)
        main_index = _read_main_index(handle, index_idx, int(qi_index["nrEntries"]))
        info_guids = _read_info_guids(handle, static_packets, main_index)
        dynamic_packets = _read_dynamic_packets(handle, static_packets, main_index)
        patient_info = _read_patient_info(handle, static_packets, main_index)
        sig_info = _read_signal_info(handle, static_packets, main_index)
        channel_info = _read_channel_info(handle, static_packets, main_index)
        montage_info = _read_montage_info(handle, static_packets, main_index)
        montage_info2 = _read_dynamic_montages(dynamic_packets)
        ts_packets = _read_tsinfo_packets(handle, static_packets, dynamic_packets, main_index)
        segments = _read_segments(handle, static_packets, main_index, ts_packets)
        events = _read_events(handle, static_packets, main_index, segments)
        event_type_info = _read_event_type_info(
            handle,
            static_packets,
            main_index,
            target_guids={event.GUID for event in events},
        )
        if event_type_info:
            for event in events:
                if event.label == "-":
                    event.label = ""
                if event.GUID in event_type_info:
                    if event.IDStr == "UNKNOWN":
                        event.IDStr = event_type_info[event.GUID]
                    if not event.label:
                        event.label = event_type_info[event.GUID]
        for event in events:
            if event.label and not any(ch.isascii() and ch.isalnum() for ch in event.label):
                event.label = None
            if event.IDStr == "Impedanse":
                label_text = (event.label or "").lower()
                if "impedance" in label_text or "impedanse" in label_text:
                    event.label = None
            if event.IDStr == "Funn av langsom aktivitet":
                event.IDStr = "Funn: langsom aktivitet"
            if event.IDStr in {"Detections Active", "Detections Inactive"} and event.annotation:
                event.label = f"{event.IDStr} - {event.annotation}"
                event.annotation = None
            if event.IDStr == "Funn" and event.label and _looks_like_channel_label(event.label):
                event.label = None
            if event.IDStr in {"UTBRUDD", "Utbrudd"}:
                event.IDStr = "Utbrudd"
                if event.label and _looks_like_channel_label(event.label):
                    event.label = None
            if event.IDStr == "SW" and event.label and _looks_like_channel_label(event.label):
                event.label = None
            if event.IDStr == "Prune" and event.label:
                lowered = event.label.lower()
                if _looks_like_channel_label(event.label) or "marks epochs" in lowered:
                    event.label = None
            if event.label and event.IDStr:
                if event.label in event.IDStr and len(event.label) < len(event.IDStr):
                    event.label = None
                elif _clean_event_label(event.label).lower() == _clean_event_label(event.IDStr).lower():
                    event.label = None
                elif event.IDStr.startswith("Funn:") and event.label.startswith("Funn av"):
                    event.label = None
            if event.IDStr == "Seizure" and (not event.label or "Anfall" in event.label):
                event.IDStr = "Anfall"
            if event.IDStr == "Annotation" and not event.annotation:
                cleaned = event.label.strip() if event.label else ""
                if not cleaned or cleaned == "-":
                    event.IDStr = "Anmerkning"
        for event in events:
            if event.IDStr == "UNKNOWN":
                cleaned = _clean_event_label(event.label or "")
                if cleaned and cleaned != "-":
                    event.IDStr = cleaned
                else:
                    raw = event.label.strip() if event.label else ""
                    if raw and raw != "-":
                        event.IDStr = raw

    nrv_header = NervusHeader(
        filename=filename,
        StaticPackets=static_packets,
        QIIndex=qi_index,
        QIIndex2=qi_index2,
        MainIndex=main_index,
        allIndexIDs=[entry.sectionIdx for entry in main_index],
        infoGuids=info_guids,
        DynamicPackets=dynamic_packets,
        PatientInfo=patient_info,
        SigInfo=sig_info,
        ChannelInfo=channel_info,
        TSInfo=ts_packets[0]["entries"] if ts_packets else [],
        TSInfoBySegment=[_select_tsinfo_for_segment(seg.date, ts_packets) for seg in segments],
        Segments=segments,
        Events=events,
        MontageInfo=montage_info,
        MontageInfo2=montage_info2,
        EventTypeInfo=event_type_info,
        format="nicolet-e",
    )

    public_header = _build_public_header(nrv_header)
    return public_header, nrv_header
